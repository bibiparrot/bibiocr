use std::{
    net::IpAddr,
    sync::{Mutex, MutexGuard},
    time::Duration,
};

use reqwest::Url;

const MIB: u64 = 1024 * 1024;
const MIB_F64: f64 = 1_048_576.0;
const SAMPLE_REQUEST_BYTES: u64 = 16 * MIB;
const UNEVEN_REQUEST_BYTES: u64 = 32 * MIB;
const DEGRADED_AGGREGATE_MIB_PER_SECOND: f64 = 32.0;
const UNEVEN_RATE_RATIO: f64 = 2.0;
const HIGH_LATENCY: Duration = Duration::from_millis(250);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NetworkProfile {
    Manual,
    Local,
    Sampling,
    Stable,
    Degraded,
    Uneven,
}

impl NetworkProfile {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Manual => "manual",
            Self::Local => "local",
            Self::Sampling => "sampling",
            Self::Stable => "stable",
            Self::Degraded => "degraded",
            Self::Uneven => "uneven",
        }
    }
}

#[derive(Debug)]
struct RouterState {
    profile: NetworkProfile,
    worker_rates: Vec<Option<f64>>,
    classified: bool,
    updates_since_classification: usize,
    aggregate_mib_per_second: Option<f64>,
    rate_ratio: Option<f64>,
}

#[derive(Debug)]
pub(crate) struct SmartRouter {
    state: Mutex<RouterState>,
    workers: usize,
    chunk_size: u64,
    configured_max_chunks: usize,
    probe_latency: Duration,
}

impl SmartRouter {
    pub(crate) fn new(
        url: &Url,
        probe_latency: Duration,
        workers: usize,
        chunk_size: u64,
        max_request_size: u64,
        adaptive: bool,
    ) -> Self {
        let workers = workers.max(1);
        let profile = if !adaptive {
            NetworkProfile::Manual
        } else if is_local_resource(url) && probe_latency <= Duration::from_millis(8) {
            NetworkProfile::Local
        } else {
            NetworkProfile::Sampling
        };
        Self {
            state: Mutex::new(RouterState {
                profile,
                worker_rates: vec![None; workers],
                classified: matches!(profile, NetworkProfile::Manual | NetworkProfile::Local),
                updates_since_classification: 0,
                aggregate_mib_per_second: None,
                rate_ratio: None,
            }),
            workers,
            chunk_size,
            configured_max_chunks: usize::try_from(max_request_size / chunk_size)
                .unwrap_or(usize::MAX)
                .max(1),
            probe_latency,
        }
    }

    pub(crate) fn span_chunks(&self, pending_chunks: usize) -> usize {
        let profile = self.lock_state().profile;
        let (jobs_per_worker, profile_cap) = match profile {
            NetworkProfile::Manual
            | NetworkProfile::Local
            | NetworkProfile::Stable
            | NetworkProfile::Degraded => (2, self.configured_max_chunks),
            NetworkProfile::Sampling => (4, self.byte_cap_chunks(SAMPLE_REQUEST_BYTES)),
            NetworkProfile::Uneven => (4, self.byte_cap_chunks(UNEVEN_REQUEST_BYTES)),
        };
        let jobs = self.workers.saturating_mul(jobs_per_worker).max(1);
        let balanced = pending_chunks.div_ceil(jobs).max(1);
        profile_cap.min(balanced).max(1)
    }

    #[allow(clippy::cast_precision_loss)] // Sampling telemetry only; byte offsets remain exact.
    pub(crate) fn record_sample(&self, worker: usize, bytes: u64, elapsed: Duration) {
        if bytes == 0 || elapsed.is_zero() || worker >= self.workers {
            return;
        }
        let rate = bytes as f64 / MIB_F64 / elapsed.as_secs_f64();
        let mut state = self.lock_state();
        if matches!(
            state.profile,
            NetworkProfile::Manual | NetworkProfile::Local
        ) {
            return;
        }
        state.worker_rates[worker] = Some(rate);
        if !state.classified {
            if state.worker_rates.iter().all(Option::is_some) {
                classify(&mut state, self.probe_latency);
            }
            return;
        }
        state.updates_since_classification += 1;
        if state.updates_since_classification >= self.workers {
            classify(&mut state, self.probe_latency);
        }
    }

    pub(crate) fn summary(&self) -> String {
        let state = self.lock_state();
        match (state.aggregate_mib_per_second, state.rate_ratio) {
            (Some(aggregate), Some(ratio)) => format!(
                "{} (sampled aggregate {aggregate:.2} MiB/s, fastest/slowest {ratio:.2}x)",
                state.profile.label()
            ),
            _ => format!(
                "{} (probe {:.1} ms)",
                state.profile.label(),
                self.probe_latency.as_secs_f64() * 1000.0
            ),
        }
    }

    fn byte_cap_chunks(&self, byte_cap: u64) -> usize {
        usize::try_from(byte_cap / self.chunk_size)
            .unwrap_or(usize::MAX)
            .max(1)
            .min(self.configured_max_chunks)
    }

    fn lock_state(&self) -> MutexGuard<'_, RouterState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

fn classify(state: &mut RouterState, probe_latency: Duration) {
    let rates = state
        .worker_rates
        .iter()
        .copied()
        .flatten()
        .collect::<Vec<_>>();
    if rates.is_empty() {
        return;
    }
    let aggregate = rates.iter().sum::<f64>();
    let slowest = rates.iter().copied().fold(f64::INFINITY, f64::min);
    let fastest = rates.iter().copied().fold(0.0_f64, f64::max);
    let ratio = fastest / slowest.max(f64::EPSILON);
    state.profile = if ratio >= UNEVEN_RATE_RATIO {
        NetworkProfile::Uneven
    } else if aggregate < DEGRADED_AGGREGATE_MIB_PER_SECOND || probe_latency >= HIGH_LATENCY {
        NetworkProfile::Degraded
    } else {
        NetworkProfile::Stable
    };
    state.classified = true;
    state.updates_since_classification = 0;
    state.aggregate_mib_per_second = Some(aggregate);
    state.rate_ratio = Some(ratio);
}

fn is_local_resource(url: &Url) -> bool {
    let Some(host) = url.host_str() else {
        return false;
    };
    if host.eq_ignore_ascii_case("localhost") || host.ends_with(".localhost") {
        return true;
    }
    host.parse::<IpAddr>().is_ok_and(|address| match address {
        IpAddr::V4(address) => {
            address.is_loopback() || address.is_private() || address.is_link_local()
        }
        IpAddr::V6(address) => address.is_loopback() || address.is_unique_local(),
    })
}

#[cfg(test)]
mod tests {
    use super::{NetworkProfile, SmartRouter};
    use reqwest::Url;
    use std::time::Duration;

    fn router(url: &str) -> SmartRouter {
        SmartRouter::new(
            &Url::parse(url).unwrap(),
            Duration::from_millis(80),
            4,
            4 * 1024 * 1024,
            128 * 1024 * 1024,
            true,
        )
    }

    #[test]
    fn private_hosts_use_the_local_profile() {
        let router = SmartRouter::new(
            &Url::parse("http://192.168.1.10/file").unwrap(),
            Duration::from_millis(2),
            4,
            4 * 1024 * 1024,
            128 * 1024 * 1024,
            true,
        );
        assert!(router.summary().starts_with(NetworkProfile::Local.label()));
    }

    #[test]
    fn explicit_request_size_disables_profile_changes() {
        let router = SmartRouter::new(
            &Url::parse("https://cdn.example/file").unwrap(),
            Duration::from_millis(80),
            4,
            4 * 1024 * 1024,
            64 * 1024 * 1024,
            false,
        );
        for worker in 0..4 {
            router.record_sample(worker, 16 * 1024 * 1024, Duration::from_secs(8));
        }
        assert!(router.summary().starts_with(NetworkProfile::Manual.label()));
    }

    #[test]
    fn uniform_slow_workers_select_degraded_long_ranges() {
        let router = router("https://cdn.example/file");
        for worker in 0..4 {
            router.record_sample(worker, 16 * 1024 * 1024, Duration::from_secs(8));
        }
        assert!(
            router
                .summary()
                .starts_with(NetworkProfile::Degraded.label())
        );
        assert_eq!(router.span_chunks(128), 16);
    }

    #[test]
    fn uneven_workers_select_shorter_balanced_ranges() {
        let router = router("https://cdn.example/file");
        for (worker, seconds) in [1, 1, 4, 4].into_iter().enumerate() {
            router.record_sample(worker, 16 * 1024 * 1024, Duration::from_secs(seconds));
        }
        assert!(router.summary().starts_with(NetworkProfile::Uneven.label()));
        assert_eq!(router.span_chunks(128), 8);
    }

    #[test]
    fn fast_uniform_workers_select_stable_ranges() {
        let router = router("https://cdn.example/file");
        for worker in 0..4 {
            router.record_sample(worker, 16 * 1024 * 1024, Duration::from_millis(500));
        }
        assert!(router.summary().starts_with(NetworkProfile::Stable.label()));
        assert_eq!(router.span_chunks(128), 16);
    }
}
