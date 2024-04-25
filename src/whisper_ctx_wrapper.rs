use std::sync::Arc;

use crate::{WhisperContext, WhisperContextParameters, WhisperError, WhisperState};

pub struct WhisperContextWrapper {
    ctx: Arc<WhisperContext>,
}

impl WhisperContextWrapper {
    /// wrapper of WhisperContext::new_with_params.
    pub fn new_with_params(
        path: &str,
        parameters: WhisperContextParameters,
    ) -> Result<Self, WhisperError> {
        let ctx = WhisperContext::new_with_params(path, parameters)?;
        Ok(Self { ctx: Arc::new(ctx) })
    }

    // we don't implement `whisper_init()` here since i have zero clue what `whisper_model_loader` does

    /// Create a new state object, ready for use.
    ///
    /// # Returns
    /// Ok(WhisperState) on success, Err(WhisperError) on failure.
    ///
    /// # C++ equivalent
    /// `struct whisper_state * whisper_init_state(struct whisper_context * ctx);`
    pub fn create_state(&self) -> Result<WhisperState, WhisperError> {
        let state = unsafe { whisper_rs_sys::whisper_init_state(self.ctx.ctx) };
        if state.is_null() {
            Err(WhisperError::InitError)
        } else {
            // SAFETY: this is known to be a valid pointer to a `whisper_state` struct
            Ok(WhisperState::new(self.ctx.clone(), state))
        }
    }
}