use std::marker::PhantomData;

/// Rustified pointer to a Whisper state.
#[derive(Debug)]
pub struct WhisperState<'a> {
    ptr: *mut whisper_rs_sys::whisper_state,
    _phantom: PhantomData<&'a ()>,
}

unsafe impl<'a> Send for WhisperState<'a> {}
unsafe impl<'a> Sync for WhisperState<'a> {}

impl<'a> Drop for WhisperState<'a> {
    fn drop(&mut self) {
        unsafe {
            whisper_rs_sys::whisper_free_state(self.ptr);
        }
    }
}

impl<'a> WhisperState<'a> {
    pub(crate) fn new(ptr: *mut whisper_rs_sys::whisper_state) -> Self {
        Self {
            ptr,
            _phantom: PhantomData,
        }
    }

    pub(crate) fn as_ptr(&self) -> *mut whisper_rs_sys::whisper_state {
        self.ptr
    }
}
