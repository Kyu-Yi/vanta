use std::fs::OpenOptions;
use std::io;
use std::io::IsTerminal;
use std::os::fd::AsRawFd;

pub struct TerminalRestoreGuard {
    tty: Option<std::fs::File>,
    original: Option<libc::termios>,
}

impl TerminalRestoreGuard {
    pub fn new() -> Self {
        if !should_activate_restore_guard(io::stdin().is_terminal(), io::stderr().is_terminal()) {
            return Self {
                tty: None,
                original: None,
            };
        }

        let tty = OpenOptions::new()
            .read(true)
            .write(true)
            .open("/dev/tty")
            .ok();
        let original = tty
            .as_ref()
            .and_then(|tty| read_termios(tty.as_raw_fd()));

        Self { tty, original }
    }
}

impl Drop for TerminalRestoreGuard {
    fn drop(&mut self) {
        let (Some(tty), Some(original)) = (&self.tty, &self.original) else {
            return;
        };
        // Best-effort terminal repair after Shuru-backed runs. Ignore errors so
        // cleanup never obscures the real pipeline outcome.
        unsafe {
            let _ = libc::tcsetattr(tty.as_raw_fd(), libc::TCSANOW, original);
        }
    }
}

fn read_termios(fd: i32) -> Option<libc::termios> {
    let mut term = std::mem::MaybeUninit::<libc::termios>::uninit();
    let rc = unsafe { libc::tcgetattr(fd, term.as_mut_ptr()) };
    if rc == 0 {
        Some(unsafe { term.assume_init() })
    } else {
        None
    }
}

fn should_activate_restore_guard(stdin_is_terminal: bool, stderr_is_terminal: bool) -> bool {
    stdin_is_terminal || stderr_is_terminal
}

#[cfg(test)]
mod tests {
    use super::should_activate_restore_guard;

    #[test]
    fn restore_guard_activates_when_any_terminal_is_attached() {
        assert!(should_activate_restore_guard(true, false));
        assert!(should_activate_restore_guard(false, true));
    }

    #[test]
    fn restore_guard_skips_non_terminal_runs() {
        assert!(!should_activate_restore_guard(false, false));
    }
}
