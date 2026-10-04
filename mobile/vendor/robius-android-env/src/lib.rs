//! JNI access to the current Android Activity.

pub use jni::{
    JNIEnv, JavaVM,
    errors::{Error, JniError, Result},
    objects::JObject,
};

pub mod sys {
    pub use jni::sys::{JNIEnv, jobject};
}

mod ndk_context;

/// # Safety
/// `activity` must be a live JNI reference to an Android Activity for this call.
pub unsafe fn set_activity(activity: *mut std::ffi::c_void) -> Result<()> {
    // SAFETY: The caller guarantees the AndroidApp Activity reference is live.
    unsafe { ndk_context::set_activity(activity) }
}

pub fn with_activity<F, R>(f: F) -> Result<R>
where
    F: for<'a, 'b, 'c, 'd> FnOnce(&'a mut JNIEnv<'b>, &'c JObject<'d>) -> R,
{
    ndk_context::with_activity_inner(f)
}
