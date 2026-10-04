use std::{
    collections::VecDeque,
    ffi::OsString,
    fs::{File, OpenOptions},
    io::{BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering},
        mpsc::Sender,
    },
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow, bail, ensure};
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use percent_encoding::percent_decode_str;
use reqwest::{
    Client, StatusCode, Url, Version,
    header::{
        ACCEPT_ENCODING, CONTENT_LENGTH, CONTENT_RANGE, ETAG, HeaderMap, HeaderName, IF_RANGE,
        LAST_MODIFIED, RANGE,
    },
};
use serde::{Deserialize, Serialize};
use tokio::{
    sync::{Mutex, watch},
    task::JoinSet,
};

use crate::router::SmartRouter;

const STATE_VERSION: u8 = 1;
const CHECKPOINT_INTERVAL: Duration = Duration::from_secs(5);
const AUTO_REQUEST_SIZE: u64 = 128 * 1024 * 1024;

#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)] // Mirrors upstream command options.
pub struct DownloadOptions {
    pub urls: Vec<Url>,
    pub output: Option<PathBuf>,
    pub max_speed: u64,
    pub num_connections: usize,
    pub headers: HeaderMap,
    pub user_agent: String,
    pub no_proxy: bool,
    pub quiet: bool,
    pub verbose: u8,
    pub alternate: bool,
    pub timeout: Duration,
    pub chunk_size: u64,
    pub request_size: Option<u64>,
    pub retries: usize,
    pub resume: bool,
    pub proxy: Option<String>,
    pub progress_events: Option<Sender<(u64, u64)>>,
}

#[derive(Debug, Clone)]
pub struct DownloadReport {
    pub output: PathBuf,
    pub bytes: u64,
    pub transferred_bytes: u64,
    pub elapsed: Duration,
    pub resumed: bool,
    pub quiet: bool,
}

#[derive(Debug, Clone)]
struct Source {
    url: Url,
    total: Option<u64>,
    range_supported: bool,
    etag: Option<String>,
    last_modified: Option<String>,
    version: Version,
    probe_latency: Duration,
    failures: Arc<AtomicU32>,
}

impl Source {
    fn validator(&self) -> Option<&str> {
        self.etag.as_deref().or(self.last_modified.as_deref())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResumeState {
    version: u8,
    generation: u64,
    primary_url: String,
    total: u64,
    chunk_size: u64,
    etag: Option<String>,
    last_modified: Option<String>,
    completed: Vec<bool>,
}

impl ResumeState {
    fn fresh(source: &Source, total: u64, chunk_size: u64) -> Self {
        let chunks = total.div_ceil(chunk_size);
        Self {
            version: STATE_VERSION,
            generation: 0,
            primary_url: resource_identity(&source.url, source.etag.is_some()),
            total,
            chunk_size,
            etag: source.etag.clone(),
            last_modified: source.last_modified.clone(),
            completed: vec![false; usize::try_from(chunks).expect("chunk count fits usize")],
        }
    }

    fn is_compatible(&self, source: &Source, total: u64, chunk_size: u64) -> bool {
        let matching_strong_etag = self
            .etag
            .as_ref()
            .is_some_and(|saved| source.etag.as_ref() == Some(saved));
        if self.version != STATE_VERSION
            || resource_identity_text(&self.primary_url, matching_strong_etag)
                != resource_identity(&source.url, matching_strong_etag)
            || self.total != total
            || self.chunk_size != chunk_size
            || self.completed.len() != usize::try_from(total.div_ceil(chunk_size)).unwrap_or(0)
        {
            return false;
        }

        match (&self.etag, &self.last_modified) {
            (Some(_), _) => matching_strong_etag,
            (None, Some(saved)) => source.last_modified.as_ref() == Some(saved),
            (None, None) => false,
        }
    }

    fn completed_bytes(&self) -> u64 {
        self.completed
            .iter()
            .enumerate()
            .filter(|(_, done)| **done)
            .map(|(index, _)| chunk_len(index, self.total, self.chunk_size))
            .sum()
    }
}

#[derive(Debug)]
struct BandwidthLimiter {
    max_speed: u64,
    start: Instant,
    granted: AtomicU64,
}

impl BandwidthLimiter {
    fn new(max_speed: u64) -> Self {
        Self {
            max_speed,
            start: Instant::now(),
            granted: AtomicU64::new(0),
        }
    }

    async fn acquire(&self, bytes: u64) {
        if self.max_speed == 0 {
            return;
        }
        let granted = self.granted.fetch_add(bytes, Ordering::Relaxed) + bytes;
        let target_nanos = u128::from(granted) * 1_000_000_000 / u128::from(self.max_speed);
        let target = Duration::from_nanos(u64::try_from(target_nanos).unwrap_or(u64::MAX));
        if let Some(delay) = target.checked_sub(self.start.elapsed()) {
            tokio::time::sleep(delay).await;
        }
    }
}

/// Downloads a remote object and atomically publishes it at the selected path.
///
/// # Errors
///
/// Returns an error for invalid server responses, exhausted retries, resume
/// state or filesystem failures, and user interruption.
pub async fn download(options: DownloadOptions) -> Result<DownloadReport> {
    let started = Instant::now();
    let client = build_client(&options)?;
    let mut sources = Vec::with_capacity(options.urls.len());
    for url in &options.urls {
        match probe(&client, url.clone()).await {
            Ok(source) => sources.push(source),
            Err(error) if !sources.is_empty() => {
                if options.verbose > 0 {
                    eprintln!("Skipping mirror {url}: {error:#}");
                }
            }
            Err(error) => return Err(error),
        }
    }

    let primary = sources
        .first()
        .ok_or_else(|| anyhow!("no usable download source"))?
        .clone();
    let output = choose_output(options.output.as_deref(), &primary.url);
    let total = primary
        .total
        .ok_or_else(|| anyhow!("server did not provide a usable file size"))?;

    sources.retain(|source| {
        source.total == Some(total)
            && source.range_supported == primary.range_supported
            && compatible_validator(&primary, source)
    });
    if sources.is_empty() {
        bail!("no mirrors agree with the primary file size and range behavior");
    }

    if options.verbose > 0 {
        eprintln!(
            "Downloading {total} bytes from {} source(s) with up to {} connection(s)",
            sources.len(),
            options.num_connections
        );
        eprintln!(
            "Negotiated {:?}; transfer ranges are up to {} bytes",
            primary.version,
            request_cap(&options)
        );
    }

    let (bytes, transferred_bytes, resumed) = if primary.range_supported {
        download_segmented(&client, &sources, &output, total, &options).await?
    } else {
        if options.verbose > 0 {
            eprintln!("Server ignored byte ranges; using one sequential connection");
        }
        download_sequential(&client, &primary, &output, total, &options).await?;
        (total, total, false)
    };

    Ok(DownloadReport {
        output,
        bytes,
        transferred_bytes,
        elapsed: started.elapsed(),
        resumed,
        quiet: options.quiet,
    })
}

fn build_client(options: &DownloadOptions) -> Result<Client> {
    let mut builder = Client::builder()
        .default_headers(options.headers.clone())
        .user_agent(options.user_agent.clone())
        .connect_timeout(options.timeout.min(Duration::from_secs(15)))
        .read_timeout(options.timeout)
        .tcp_keepalive(Duration::from_secs(30))
        .pool_max_idle_per_host(options.num_connections.max(1));
    if options.num_connections > 1 {
        // `-n` promises concurrent connections, not merely multiplexed streams on
        // one HTTP/2 transport. HTTP/1.1 makes the connection count effective on
        // origins that throttle each TCP flow independently.
        builder = builder.http1_only();
    }
    if options.no_proxy {
        builder = builder.no_proxy();
    } else if let Some(proxy) = options.proxy.as_deref() {
        builder = builder.proxy(reqwest::Proxy::all(proxy).context("invalid proxy")?);
    }
    builder.build().context("failed to initialize HTTP client")
}

async fn probe(client: &Client, url: Url) -> Result<Source> {
    let probe_started = Instant::now();
    let response = client
        .get(url.clone())
        .header(RANGE, "bytes=0-0")
        .header(ACCEPT_ENCODING, "identity")
        .send()
        .await
        .with_context(|| format!("failed to probe {url}"))?;

    let status = response.status();
    let version = response.version();
    let final_url = response.url().clone();
    ensure!(
        status.is_success(),
        "probe for {url} returned HTTP {status}"
    );
    let headers = response.headers();
    let etag = strong_etag(headers);
    let last_modified = header_text(headers, LAST_MODIFIED);

    let (total, range_supported) = if status == StatusCode::PARTIAL_CONTENT {
        let value = headers
            .get(CONTENT_RANGE)
            .and_then(|value| value.to_str().ok())
            .ok_or_else(|| anyhow!("206 response from {url} lacks Content-Range"))?;
        let (start, end, total) = parse_content_range(value)?;
        ensure!(start == 0 && end == 0, "unexpected probe range: {value}");
        (Some(total), true)
    } else {
        (response.content_length(), false)
    };

    if range_supported {
        let body = response
            .bytes()
            .await
            .with_context(|| format!("failed to read range probe body from {final_url}"))?;
        ensure!(
            body.len() == 1,
            "range probe returned {} body bytes",
            body.len()
        );
    }

    Ok(Source {
        // Keep the caller's URL for every transfer. Letting reqwest follow each
        // redirect preserves its cross-origin stripping of sensitive headers and
        // allows redirect-mediated authentication to refresh normally.
        url,
        total,
        range_supported,
        etag,
        last_modified,
        version,
        probe_latency: probe_started.elapsed(),
        failures: Arc::new(AtomicU32::new(0)),
    })
}

fn resource_identity(url: &Url, redact_volatile_auth: bool) -> String {
    let mut identity = url.clone();
    identity.set_fragment(None);
    if redact_volatile_auth {
        let retained = identity
            .query_pairs()
            .filter(|(name, _)| !is_volatile_auth_query_key(name))
            .map(|(name, value)| (name.into_owned(), value.into_owned()))
            .collect::<Vec<_>>();
        identity.set_query(None);
        if !retained.is_empty() {
            identity.query_pairs_mut().extend_pairs(retained);
        }
    }
    identity.to_string()
}

fn resource_identity_text(raw: &str, redact_volatile_auth: bool) -> String {
    Url::parse(raw).map_or_else(
        |_| raw.to_owned(),
        |url| resource_identity(&url, redact_volatile_auth),
    )
}

fn is_volatile_auth_query_key(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    matches!(
        name.as_str(),
        "auth_key"
            | "authkey"
            | "authorization"
            | "signature"
            | "sig"
            | "token"
            | "access_token"
            | "expires"
            | "expiry"
            | "expiration"
            | "policy"
            | "key-pair-id"
    ) || ["x-amz-", "x-goog-", "x-oss-", "x-cos-", "x-bce-"]
        .iter()
        .any(|prefix| name.starts_with(prefix))
}

fn request_cap(options: &DownloadOptions) -> u64 {
    options
        .request_size
        .unwrap_or(AUTO_REQUEST_SIZE.max(options.chunk_size))
}

fn compatible_validator(primary: &Source, candidate: &Source) -> bool {
    if let Some(etag) = &primary.etag {
        return candidate.etag.as_ref() == Some(etag);
    }
    if let Some(last_modified) = &primary.last_modified {
        return candidate.last_modified.as_ref() == Some(last_modified);
    }
    candidate.url == primary.url
}

fn strong_etag(headers: &HeaderMap) -> Option<String> {
    header_text(headers, ETAG).filter(|etag| !etag.trim_start().starts_with("W/"))
}

fn header_text(headers: &HeaderMap, name: HeaderName) -> Option<String> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(ToOwned::to_owned)
}

fn parse_content_range(value: &str) -> Result<(u64, u64, u64)> {
    let value = value
        .strip_prefix("bytes ")
        .ok_or_else(|| anyhow!("invalid Content-Range unit: {value}"))?;
    let (range, total) = value
        .split_once('/')
        .ok_or_else(|| anyhow!("invalid Content-Range: {value}"))?;
    let (start, end) = range
        .split_once('-')
        .ok_or_else(|| anyhow!("invalid Content-Range: {value}"))?;
    Ok((start.parse()?, end.parse()?, total.parse()?))
}

#[allow(clippy::too_many_lines)] // Transfer lifecycle coordinator; leaf work is delegated.
async fn download_segmented(
    client: &Client,
    sources: &[Source],
    output: &Path,
    total: u64,
    options: &DownloadOptions,
) -> Result<(u64, u64, bool)> {
    let part_path = companion_path(output, ".bibiget.part");
    if !options.resume {
        for path in state_paths(output) {
            match std::fs::remove_file(&path) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => {
                    return Err(error)
                        .with_context(|| format!("failed to remove {}", path.display()));
                }
            }
        }
    }
    let loaded = if options.resume {
        load_state(output)?
    } else {
        None
    };
    let (mut state, resumed) = match loaded {
        Some(state) if state.is_compatible(&sources[0], total, options.chunk_size) => {
            let part_len = part_path.metadata().map_or(0, |meta| meta.len());
            if part_len == total {
                (state, true)
            } else {
                eprintln!("warning: resume data length is invalid; restarting download");
                (
                    ResumeState::fresh(&sources[0], total, options.chunk_size),
                    false,
                )
            }
        }
        Some(_) => {
            eprintln!("warning: resume metadata does not match the remote object; restarting");
            (
                ResumeState::fresh(&sources[0], total, options.chunk_size),
                false,
            )
        }
        None => (
            ResumeState::fresh(&sources[0], total, options.chunk_size),
            false,
        ),
    };

    reject_existing_output(output, resumed)?;
    let file = Arc::new(open_part_file(&part_path, total, !resumed)?);
    if !resumed {
        state.completed.fill(false);
    }
    save_state(output, &mut state)?;

    let initial = state.completed_bytes();
    let progress = make_progress(total, initial, options);
    let pending = state
        .completed
        .iter()
        .enumerate()
        .filter_map(|(index, done)| (!done).then_some(index))
        .collect::<VecDeque<_>>();
    let pending_chunks = pending.len();
    let queue = Arc::new(Mutex::new(pending));
    let cancel = Arc::new(AtomicBool::new(false));
    let (interrupt_tx, interrupt_rx) = watch::channel(false);
    let limiter = Arc::new(BandwidthLimiter::new(options.max_speed));
    let (completed_tx, completed_rx) = tokio::sync::mpsc::unbounded_channel();

    let checkpoint_file = Arc::clone(&file);
    let checkpoint_output = output.to_path_buf();
    let checkpoint = tokio::spawn(async move {
        checkpoint_manager(state, completed_rx, checkpoint_file, checkpoint_output).await
    });

    let signal_cancel = Arc::clone(&cancel);
    let signal_interrupt = interrupt_tx.clone();
    let signal = tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            signal_cancel.store(true, Ordering::Release);
            let _ = signal_interrupt.send(true);
        }
    });

    let workers = options.num_connections.min(pending_chunks).max(1);
    let router = Arc::new(SmartRouter::new(
        &sources[0].url,
        sources[0].probe_latency,
        workers,
        options.chunk_size,
        request_cap(options),
        options.request_size.is_none(),
    ));
    if options.verbose > 0 {
        eprintln!("Smart router initial profile: {}", router.summary());
    }
    let mut tasks = JoinSet::new();
    for worker in 0..workers {
        let client = client.clone();
        let sources = sources.to_vec();
        let file = Arc::clone(&file);
        let queue = Arc::clone(&queue);
        let cancel = Arc::clone(&cancel);
        let limiter = Arc::clone(&limiter);
        let tx = completed_tx.clone();
        let progress = progress.clone();
        let progress_events = options.progress_events.clone();
        let router = Arc::clone(&router);
        let interrupt = interrupt_rx.clone();
        let retries = options.retries;
        let chunk_size = options.chunk_size;
        tasks.spawn(async move {
            loop {
                if cancel.load(Ordering::Acquire) {
                    break;
                }
                let Some(span) = claim_span(&queue, &router).await else {
                    break;
                };
                let task = ChunkTask {
                    client: &client,
                    sources: &sources,
                    file: &file,
                    total,
                    chunk_size,
                    retries,
                    limiter: &limiter,
                    worker,
                    completed_tx: &tx,
                    progress: &progress,
                    progress_events: progress_events.clone(),
                    interrupt: interrupt.clone(),
                };
                let span_bytes = span
                    .iter()
                    .map(|&index| chunk_len(index, total, chunk_size))
                    .sum();
                let span_started = Instant::now();
                let result = download_span_with_retry(&task, &span).await;
                if let Err(error) = result {
                    cancel.store(true, Ordering::Release);
                    return Err(error);
                }
                router.record_sample(worker, span_bytes, span_started.elapsed());
            }
            Ok(())
        });
    }
    drop(completed_tx);

    let mut worker_error = None;
    while let Some(result) = tasks.join_next().await {
        match result {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                worker_error.get_or_insert(error);
            }
            Err(error) => {
                worker_error.get_or_insert_with(|| anyhow!("download worker failed: {error}"));
            }
        }
    }
    signal.abort();
    drop(interrupt_tx);

    let final_state = checkpoint
        .await
        .context("resume checkpoint task panicked")??;
    progress.finish_and_clear();

    if options.verbose > 0 {
        eprintln!("Smart router final profile: {}", router.summary());
    }

    if let Some(error) = worker_error {
        return Err(error);
    }
    ensure!(
        final_state.completed.iter().all(|done| *done),
        "download interrupted; resume data was saved"
    );

    let finish_file = Arc::clone(&file);
    tokio::task::spawn_blocking(move || finish_file.sync_all())
        .await
        .context("final file sync task failed")??;
    drop(file);
    finish_download(output, &part_path)?;
    Ok((total, total - initial, resumed))
}

async fn claim_span(queue: &Mutex<VecDeque<usize>>, router: &SmartRouter) -> Option<Vec<usize>> {
    let mut queue = queue.lock().await;
    let max_chunks = router.span_chunks(queue.len());
    let first = queue.pop_front()?;
    let mut span = Vec::with_capacity(max_chunks.min(queue.len() + 1));
    span.push(first);
    while span.len() < max_chunks {
        let Some(&next) = queue.front() else {
            break;
        };
        if next != span.last().copied().expect("span is nonempty") + 1 {
            break;
        }
        span.push(queue.pop_front().expect("front was present"));
    }
    Some(span)
}

struct ChunkTask<'a> {
    client: &'a Client,
    sources: &'a [Source],
    file: &'a File,
    total: u64,
    chunk_size: u64,
    retries: usize,
    limiter: &'a BandwidthLimiter,
    worker: usize,
    completed_tx: &'a tokio::sync::mpsc::UnboundedSender<usize>,
    progress: &'a ProgressBar,
    progress_events: Option<Sender<(u64, u64)>>,
    interrupt: watch::Receiver<bool>,
}

async fn download_span_with_retry(task: &ChunkTask<'_>, span: &[usize]) -> Result<()> {
    let first = *span.first().ok_or_else(|| anyhow!("empty chunk span"))?;
    let mut last_error = None;
    let mut received = 0;
    for attempt in 0..task.retries {
        let preferred = (first + attempt + task.worker) % task.sources.len();
        let source = &task.sources[healthiest_source(task.sources, preferred)];
        let (new_received, result) = download_span(task, source, span, received).await;
        received = new_received;
        match result {
            Ok(()) => {
                source.failures.store(0, Ordering::Relaxed);
                return Ok(());
            }
            Err(error) => {
                if *task.interrupt.borrow() {
                    return Err(interrupted_error());
                }
                source.failures.fetch_add(1, Ordering::Relaxed);
                last_error = Some(error);
                if attempt + 1 < task.retries {
                    let delay =
                        Duration::from_millis(100_u64.saturating_mul(1_u64 << attempt.min(8)));
                    let mut interrupt = task.interrupt.clone();
                    tokio::select! {
                        biased;
                        () = wait_until_interrupted(&mut interrupt) => {
                            return Err(interrupted_error());
                        }
                        () = tokio::time::sleep(delay) => {}
                    }
                }
            }
        }
    }
    Err(last_error.unwrap_or_else(|| anyhow!("chunk span at {first} failed without an error")))
        .with_context(|| {
            format!(
                "chunk span at {first} exhausted {} attempt(s)",
                task.retries
            )
        })
}

fn healthiest_source(sources: &[Source], preferred: usize) -> usize {
    let minimum = sources
        .iter()
        .map(|source| source.failures.load(Ordering::Relaxed))
        .min()
        .unwrap_or(0);
    (0..sources.len())
        .map(|offset| (preferred + offset) % sources.len())
        .find(|&index| sources[index].failures.load(Ordering::Relaxed) == minimum)
        .unwrap_or(preferred)
}

#[allow(clippy::too_many_lines)] // Keeps range validation and its streaming write loop together.
async fn download_span(
    task: &ChunkTask<'_>,
    source: &Source,
    span: &[usize],
    mut received: u64,
) -> (u64, Result<()>) {
    let Some(&first) = span.first() else {
        return (0, Err(anyhow!("empty chunk span")));
    };
    let Some(&last) = span.last() else {
        return (0, Err(anyhow!("empty chunk span")));
    };
    let start = match u64::try_from(first) {
        Ok(index) => index * task.chunk_size,
        Err(error) => return (0, Err(error.into())),
    };
    let last_start = last as u64 * task.chunk_size;
    let end = last_start + chunk_len(last, task.total, task.chunk_size) - 1;
    let length = end - start + 1;
    if received >= length {
        return (received, Ok(()));
    }
    let request_start = start + received;
    let remaining = length - received;
    let mut reported = 0;
    let mut completed_boundary = 0;
    while reported < span.len() {
        let next_boundary =
            completed_boundary + chunk_len(span[reported], task.total, task.chunk_size);
        if received < next_boundary {
            break;
        }
        completed_boundary = next_boundary;
        reported += 1;
    }
    let mut interrupt = task.interrupt.clone();
    let result = async {
        if *interrupt.borrow() {
            return Err(interrupted_error());
        }
        let mut request = task
            .client
            .get(source.url.clone())
            .header(RANGE, format!("bytes={request_start}-{end}"))
            .header(ACCEPT_ENCODING, "identity");
        if let Some(validator) = source.validator() {
            request = request.header(IF_RANGE, validator);
        }
        let response = tokio::select! {
            biased;
            () = wait_until_interrupted(&mut interrupt) => {
                return Err(interrupted_error());
            }
            response = request.send() => response
                .with_context(|| format!("request for bytes {request_start}-{end} failed"))?,
        };
        ensure!(
            response.status() == StatusCode::PARTIAL_CONTENT,
            "range {request_start}-{end} returned HTTP {}",
            response.status()
        );
        let content_range = response
            .headers()
            .get(CONTENT_RANGE)
            .and_then(|value| value.to_str().ok())
            .ok_or_else(|| anyhow!("range {request_start}-{end} lacks Content-Range"))?;
        let (actual_start, actual_end, actual_total) = parse_content_range(content_range)?;
        ensure!(
            (actual_start, actual_end, actual_total) == (request_start, end, task.total),
            "server returned mismatched Content-Range {content_range}"
        );
        if let Some(content_length) = response
            .headers()
            .get(CONTENT_LENGTH)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse::<u64>().ok())
        {
            ensure!(content_length == remaining, "mismatched Content-Length");
        }

        let mut stream = response.bytes_stream();
        loop {
            let next = tokio::select! {
                biased;
                () = wait_until_interrupted(&mut interrupt) => {
                    return Err(interrupted_error());
                }
                next = stream.next() => next,
            };
            let Some(chunk) = next else {
                break;
            };
            let chunk = chunk.context("range response body failed")?;
            let body_len = u64::try_from(chunk.len())?;
            ensure!(
                received + body_len <= length,
                "range response exceeded requested length"
            );
            tokio::select! {
                biased;
                () = wait_until_interrupted(&mut interrupt) => {
                    return Err(interrupted_error());
                }
                () = task.limiter.acquire(body_len) => {}
            }
            write_all_at(task.file, &chunk, start + received)?;
            received += body_len;
            while reported < span.len() {
                let index = span[reported];
                let next_boundary =
                    completed_boundary + chunk_len(index, task.total, task.chunk_size);
                if received < next_boundary {
                    break;
                }
                completed_boundary = next_boundary;
                reported += 1;
                task.progress
                    .inc(chunk_len(index, task.total, task.chunk_size));
                if let Some(events) = &task.progress_events {
                    let _ = events.send((task.progress.position(), task.total));
                }
                task.completed_tx
                    .send(index)
                    .map_err(|_| anyhow!("resume checkpoint task stopped unexpectedly"))?;
            }
        }
        ensure!(
            received == length,
            "range ended after {received} of {length} bytes"
        );
        Ok(())
    }
    .await;
    (received, result)
}

async fn wait_until_interrupted(interrupt: &mut watch::Receiver<bool>) {
    let _ = interrupt.wait_for(|interrupted| *interrupted).await;
}

fn interrupted_error() -> anyhow::Error {
    anyhow!("download interrupted; resume data was saved")
}

async fn checkpoint_manager(
    mut state: ResumeState,
    mut completed_rx: tokio::sync::mpsc::UnboundedReceiver<usize>,
    file: Arc<File>,
    output: PathBuf,
) -> Result<ResumeState> {
    let mut pending = Vec::new();
    let mut interval = tokio::time::interval(CHECKPOINT_INTERVAL);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    loop {
        tokio::select! {
            item = completed_rx.recv() => if let Some(index) = item {
                pending.push(index);
            } else {
                checkpoint_pending(&mut state, &mut pending, &file, &output).await?;
                return Ok(state);
            },
            _ = interval.tick(), if !pending.is_empty() => {
                checkpoint_pending(&mut state, &mut pending, &file, &output).await?;
            }
        }
    }
}

async fn checkpoint_pending(
    state: &mut ResumeState,
    pending: &mut Vec<usize>,
    file: &Arc<File>,
    output: &Path,
) -> Result<()> {
    if pending.is_empty() {
        return Ok(());
    }
    let sync_file = Arc::clone(file);
    tokio::task::spawn_blocking(move || sync_file.sync_data())
        .await
        .context("checkpoint file sync task failed")??;
    for index in pending.drain(..) {
        state.completed[index] = true;
    }
    let mut snapshot = state.clone();
    let output = output.to_path_buf();
    tokio::task::spawn_blocking(move || save_state(&output, &mut snapshot))
        .await
        .context("checkpoint metadata task failed")??;
    state.generation = state.generation.saturating_add(1);
    Ok(())
}

async fn download_sequential(
    client: &Client,
    source: &Source,
    output: &Path,
    total: u64,
    options: &DownloadOptions,
) -> Result<()> {
    reject_existing_output(output, false)?;
    let part_path = companion_path(output, ".bibiget.part");
    let progress = make_progress(total, 0, options);
    let limiter = BandwidthLimiter::new(options.max_speed);
    let mut last_error = None;

    for attempt in 0..options.retries {
        let result = async {
            let response = client
                .get(source.url.clone())
                .header(ACCEPT_ENCODING, "identity")
                .send()
                .await?
                .error_for_status()?;
            let mut file = BufWriter::new(
                OpenOptions::new()
                    .create(true)
                    .truncate(true)
                    .write(true)
                    .open(&part_path)?,
            );
            let mut received = 0_u64;
            let mut stream = response.bytes_stream();
            while let Some(chunk) = stream.next().await {
                let chunk = chunk?;
                let size = u64::try_from(chunk.len())?;
                limiter.acquire(size).await;
                file.write_all(&chunk)?;
                received += size;
                progress.set_position(received);
                if let Some(events) = &options.progress_events {
                    let _ = events.send((received, total));
                }
            }
            file.flush()?;
            file.get_ref().sync_all()?;
            ensure!(received == total, "received {received} of {total} bytes");
            drop(file);
            Result::<()>::Ok(())
        }
        .await;
        match result {
            Ok(()) => {
                progress.finish_and_clear();
                if let Some(events) = &options.progress_events {
                    let _ = events.send((total, total));
                }
                finish_download(output, &part_path)?;
                return Ok(());
            }
            Err(error) => {
                last_error = Some(error);
                progress.set_position(0);
                if let Some(events) = &options.progress_events {
                    let _ = events.send((0, total));
                }
                if attempt + 1 < options.retries {
                    tokio::time::sleep(Duration::from_millis(100 << attempt.min(8))).await;
                }
            }
        }
    }
    Err(last_error.unwrap_or_else(|| anyhow!("sequential download failed")))
}

fn open_part_file(path: &Path, total: u64, truncate: bool) -> Result<File> {
    let file = OpenOptions::new()
        .create(true)
        .truncate(truncate)
        .read(true)
        .write(true)
        .open(path)
        .with_context(|| format!("failed to open partial file {}", path.display()))?;
    file.set_len(total)?;
    Ok(file)
}

fn reject_existing_output(output: &Path, resuming: bool) -> Result<()> {
    if output.exists() && !resuming {
        bail!(
            "output file {} already exists and has no matching resume metadata",
            output.display()
        );
    }
    Ok(())
}

fn finish_download(output: &Path, part_path: &Path) -> Result<()> {
    ensure!(
        !output.exists(),
        "refusing to replace existing output {}",
        output.display()
    );
    std::fs::rename(part_path, output).with_context(|| {
        format!(
            "failed to publish {} as {}",
            part_path.display(),
            output.display()
        )
    })?;
    for slot in state_paths(output) {
        match std::fs::remove_file(&slot) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(error).with_context(|| format!("failed to remove {}", slot.display()));
            }
        }
    }
    Ok(())
}

fn make_progress(total: u64, initial: u64, options: &DownloadOptions) -> ProgressBar {
    if let Some(events) = &options.progress_events {
        let _ = events.send((initial, total));
    }
    if options.quiet {
        return ProgressBar::hidden();
    }
    let progress = ProgressBar::new(total);
    let template = if options.alternate {
        "{percent:>3}% {bytes}/{total_bytes} {bytes_per_sec} ETA {eta}"
    } else {
        "[{bar:40.cyan/blue}] {bytes}/{total_bytes} {bytes_per_sec} ETA {eta}"
    };
    progress.set_style(
        ProgressStyle::with_template(template)
            .expect("static progress template is valid")
            .progress_chars("=>-"),
    );
    progress.set_position(initial);
    progress
}

fn choose_output(requested: Option<&Path>, url: &Url) -> PathBuf {
    let filename = url
        .path_segments()
        .and_then(Iterator::last)
        .filter(|name| !name.is_empty())
        .map_or_else(|| "index.html".to_owned(), safe_filename);

    if let Some(requested) = requested {
        return if requested.is_dir() {
            requested.join(filename)
        } else {
            requested.to_path_buf()
        };
    }

    let base = PathBuf::from(&filename);
    if !base.exists() || state_paths(&base).iter().any(|path| path.exists()) {
        return base;
    }
    for suffix in 0_u32.. {
        let candidate = PathBuf::from(format!("{filename}.{suffix}"));
        if !candidate.exists() && !companion_path(&candidate, ".bibiget.part").exists() {
            return candidate;
        }
    }
    unreachable!("u32 filename suffix space exhausted")
}

fn safe_filename(encoded: &str) -> String {
    let decoded = percent_decode_str(encoded).decode_utf8_lossy();
    let sanitized: String = decoded
        .chars()
        .map(|character| {
            if character.is_control() || r#"/\:*?"<>|"#.contains(character) {
                '_'
            } else {
                character
            }
        })
        .collect();
    if sanitized.is_empty() || matches!(sanitized.as_str(), "." | "..") {
        "index.html".to_owned()
    } else {
        sanitized
    }
}

fn chunk_len(index: usize, total: u64, chunk_size: u64) -> u64 {
    let start = index as u64 * chunk_size;
    chunk_size.min(total - start)
}

fn companion_path(base: &Path, suffix: &str) -> PathBuf {
    let mut name: OsString = base.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

fn state_paths(output: &Path) -> [PathBuf; 2] {
    [
        companion_path(output, ".bibiget.state.0"),
        companion_path(output, ".bibiget.state.1"),
    ]
}

fn load_state(output: &Path) -> Result<Option<ResumeState>> {
    let mut valid = Vec::new();
    for path in state_paths(output) {
        match File::open(&path) {
            Ok(file) => match serde_json::from_reader::<_, ResumeState>(BufReader::new(file)) {
                Ok(state) => valid.push(state),
                Err(error) => eprintln!(
                    "warning: ignoring corrupt resume slot {}: {error}",
                    path.display()
                ),
            },
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(error).with_context(|| format!("failed to read {}", path.display()));
            }
        }
    }
    Ok(valid.into_iter().max_by_key(|state| state.generation))
}

fn save_state(output: &Path, state: &mut ResumeState) -> Result<()> {
    state.generation = state.generation.saturating_add(1);
    let slot = usize::try_from(state.generation % 2).expect("state slot is zero or one");
    let path = &state_paths(output)[slot];
    let file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(path)
        .with_context(|| format!("failed to write resume slot {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, state)?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    Ok(())
}

#[cfg(windows)]
fn write_all_at(file: &File, mut buffer: &[u8], mut offset: u64) -> Result<()> {
    use std::os::windows::fs::FileExt;
    while !buffer.is_empty() {
        let written = file.seek_write(buffer, offset)?;
        ensure!(written > 0, "positional write made no progress");
        offset += u64::try_from(written)?;
        buffer = &buffer[written..];
    }
    Ok(())
}

#[cfg(unix)]
fn write_all_at(file: &File, mut buffer: &[u8], mut offset: u64) -> Result<()> {
    use std::os::unix::fs::FileExt;
    while !buffer.is_empty() {
        let written = file.write_at(buffer, offset)?;
        ensure!(written > 0, "positional write made no progress");
        offset += u64::try_from(written)?;
        buffer = &buffer[written..];
    }
    Ok(())
}

#[cfg(not(any(unix, windows)))]
compile_error!("bibiget positional file writes require Unix or Windows");

#[cfg(test)]
mod tests {
    use std::{
        sync::{Arc, atomic::AtomicU32},
        time::Duration,
    };

    use super::{
        DownloadOptions, ResumeState, Source, build_client, chunk_len, compatible_validator,
        load_state, make_progress, parse_content_range, resource_identity, safe_filename,
        save_state, state_paths, wait_until_interrupted,
    };
    use reqwest::{Url, Version};
    use tokio::sync::watch;

    fn source(etag: Option<&str>) -> Source {
        Source {
            url: Url::parse("https://example.test/file").unwrap(),
            total: Some(10),
            range_supported: true,
            etag: etag.map(ToOwned::to_owned),
            last_modified: None,
            version: Version::HTTP_11,
            probe_latency: Duration::ZERO,
            failures: Arc::new(AtomicU32::new(0)),
        }
    }

    #[test]
    fn gui_options_apply_proxy_and_report_initial_progress() {
        let (sender, receiver) = std::sync::mpsc::channel();
        let options = DownloadOptions {
            urls: vec![Url::parse("https://example.test/file").unwrap()],
            output: None,
            max_speed: 0,
            num_connections: 8,
            headers: reqwest::header::HeaderMap::new(),
            user_agent: "bibiocr-test".to_owned(),
            no_proxy: false,
            quiet: true,
            verbose: 0,
            alternate: false,
            timeout: Duration::from_secs(1),
            chunk_size: 4 * 1024 * 1024,
            request_size: None,
            retries: 1,
            resume: true,
            proxy: Some("http://127.0.0.1:7890".to_owned()),
            progress_events: Some(sender),
        };
        build_client(&options).unwrap();
        let _ = make_progress(10, 4, &options);
        assert_eq!(receiver.recv().unwrap(), (4, 10));
    }

    #[test]
    fn parses_http_content_range() {
        assert_eq!(parse_content_range("bytes 4-7/10").unwrap(), (4, 7, 10));
        assert!(parse_content_range("items 4-7/10").is_err());
    }

    #[test]
    fn tail_chunk_has_exact_remaining_length() {
        assert_eq!(chunk_len(0, 10, 4), 4);
        assert_eq!(chunk_len(2, 10, 4), 2);
    }

    #[test]
    fn resume_requires_matching_remote_validator() {
        let state = ResumeState::fresh(&source(Some("\"v1\"")), 10, 4);
        assert!(state.is_compatible(&source(Some("\"v1\"")), 10, 4));
        assert!(!state.is_compatible(&source(Some("\"v2\"")), 10, 4));
        assert!(!ResumeState::fresh(&source(None), 10, 4).is_compatible(&source(None), 10, 4));
    }

    #[test]
    fn resume_accepts_rotated_query_token_for_same_validated_resource() {
        let mut original = source(Some("\"v1\""));
        original.url =
            Url::parse("https://example.test/file?id=model-a&auth_key=old-secret").unwrap();
        let mut state = ResumeState::fresh(&original, 10, 4);
        // Compatibility with pre-0.1.1 state that stored the full signed URL.
        state.primary_url = original.url.to_string();
        let mut refreshed = source(Some("\"v1\""));
        refreshed.url =
            Url::parse("https://example.test/file?id=model-a&auth_key=new-secret").unwrap();

        assert!(state.is_compatible(&refreshed, 10, 4));
        assert_eq!(
            resource_identity(&refreshed.url, true),
            "https://example.test/file?id=model-a"
        );
    }

    #[test]
    fn resume_identity_preserves_resource_query_parameters() {
        let mut original = source(Some("\"shared-validator\""));
        original.url =
            Url::parse("https://example.test/file?id=model-a&auth_key=old-secret").unwrap();
        let state = ResumeState::fresh(&original, 10, 4);
        let mut different = source(Some("\"shared-validator\""));
        different.url =
            Url::parse("https://example.test/file?id=model-b&auth_key=new-secret").unwrap();

        assert!(!state.is_compatible(&different, 10, 4));
    }

    #[test]
    fn token_rotation_requires_a_strong_etag() {
        let mut original = source(None);
        original.last_modified = Some("Sat, 30 Aug 2026 00:00:00 GMT".to_owned());
        original.url = Url::parse("https://example.test/file?auth_key=old-secret").unwrap();
        let state = ResumeState::fresh(&original, 10, 4);
        let mut refreshed = original.clone();
        refreshed.url = Url::parse("https://example.test/file?auth_key=new-secret").unwrap();

        assert!(!state.is_compatible(&refreshed, 10, 4));
    }

    #[test]
    fn a_new_etag_cannot_relax_a_last_modified_checkpoint_identity() {
        let mut original = source(None);
        original.last_modified = Some("Sat, 30 Aug 2026 00:00:00 GMT".to_owned());
        original.url = Url::parse("https://example.test/file?auth_key=old-secret").unwrap();
        let state = ResumeState::fresh(&original, 10, 4);
        let mut refreshed = original.clone();
        refreshed.etag = Some("\"new-strong-validator\"".to_owned());
        refreshed.url = Url::parse("https://example.test/file?auth_key=new-secret").unwrap();

        assert!(!state.is_compatible(&refreshed, 10, 4));
    }

    #[tokio::test]
    async fn interrupt_wait_observes_an_already_sent_signal() {
        let (sender, mut receiver) = watch::channel(false);
        sender.send(true).unwrap();
        tokio::time::timeout(
            Duration::from_millis(100),
            wait_until_interrupted(&mut receiver),
        )
        .await
        .expect("an interrupt sent before polling must not be lost");
    }

    #[test]
    fn mirrors_must_share_the_primary_validator() {
        let primary = source(Some("\"v1\""));
        assert!(compatible_validator(&primary, &source(Some("\"v1\""))));
        assert!(!compatible_validator(&primary, &source(Some("\"v2\""))));
        assert!(!compatible_validator(&primary, &source(None)));
    }

    #[test]
    fn corrupt_newest_resume_slot_falls_back_to_previous_generation() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("file.bin");
        let mut state = ResumeState::fresh(&source(Some("\"v1\"")), 10, 4);
        save_state(&output, &mut state).unwrap();
        state.completed[0] = true;
        save_state(&output, &mut state).unwrap();

        let newest_slot = &state_paths(&output)
            [usize::try_from(state.generation % 2).expect("state slot is zero or one")];
        std::fs::write(newest_slot, b"{torn checkpoint").unwrap();

        let recovered = load_state(&output).unwrap().expect("older valid slot");
        assert_eq!(recovered.generation + 1, state.generation);
        assert!(!recovered.completed[0]);
    }

    #[test]
    fn inferred_filename_cannot_escape_the_download_directory() {
        assert_eq!(safe_filename("%2E%2E"), "index.html");
        assert_eq!(safe_filename("folder%2Ffile.bin"), "folder_file.bin");
        assert_eq!(safe_filename("a%5Cb%3Ac"), "a_b_c");
    }
}
