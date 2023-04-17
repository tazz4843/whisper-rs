/// Rustified pointer to a Whisper state.
#[derive(Debug)]
pub struct WhisperState {
    ptr: *mut whisper_rs_sys::whisper_state,
}

unsafe impl Send for WhisperState {}
unsafe impl Sync for WhisperState {}

impl Drop for WhisperState {
    fn drop(&mut self) {
        unsafe {
            whisper_rs_sys::whisper_free_state(self.ptr);
        }
    }
}

impl WhisperState {
    pub(crate) unsafe fn new(ptr: *mut whisper_rs_sys::whisper_state) -> Self {
        Self { ptr }
    }

    pub(crate) fn as_ptr(&self) -> *mut whisper_rs_sys::whisper_state {
        self.ptr
    }
}
