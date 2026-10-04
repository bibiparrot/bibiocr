use std::sync::{Mutex, MutexGuard};

static BACKEND: Mutex<()> = Mutex::new(());

pub fn lock() -> Result<MutexGuard<'static, ()>, String> {
    BACKEND.lock().map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::lock;
    use std::{sync::mpsc, time::Duration};

    #[test]
    fn concurrent_recognitions_wait_for_the_backend() {
        let first = lock().unwrap();
        let (sender, receiver) = mpsc::channel();
        let second = std::thread::spawn(move || {
            let _guard = lock().unwrap();
            sender.send(()).unwrap();
        });
        assert!(receiver.recv_timeout(Duration::from_millis(50)).is_err());
        drop(first);
        receiver.recv_timeout(Duration::from_secs(1)).unwrap();
        second.join().unwrap();
    }
}
