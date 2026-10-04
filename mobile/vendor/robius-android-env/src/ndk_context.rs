use crate::{Error, JNIEnv, JObject, JavaVM, Result};
use jni::objects::GlobalRef;
use std::{ffi::c_void, sync::{Mutex, OnceLock}};

static ACTIVITY: OnceLock<Mutex<Option<GlobalRef>>> = OnceLock::new();

/// # Safety
/// The pointer must be a live JNI Activity reference supplied by AndroidApp.
pub unsafe fn set_activity(activity: *mut c_void) -> Result<()> {
    if activity.is_null() {
        return Err(Error::NullPtr("Android Activity"));
    }
    let context = ndk_context::android_context();
    // SAFETY: android-activity initialized ndk-context with a valid VM.
    let vm = unsafe { JavaVM::from_raw(context.vm().cast())? };
    let env = vm.attach_current_thread_permanently()?;
    // SAFETY: AndroidApp guarantees this reference is live for the call.
    let activity = unsafe { JObject::from_raw(activity.cast()) };
    let global = env.new_global_ref(&activity)?;
    *ACTIVITY.get_or_init(|| Mutex::new(None)).lock().unwrap() = Some(global);
    Ok(())
}

pub fn with_activity_inner<F, R>(f: F) -> Result<R>
where
    F: for<'a, 'b, 'c, 'd> FnOnce(&'a mut JNIEnv<'b>, &'c JObject<'d>) -> R,
{
    let activity = ACTIVITY
        .get()
        .and_then(|value| value.lock().unwrap().clone())
        .ok_or(Error::NullPtr("Android Activity not registered"))?;
    let context = ndk_context::android_context();
    // SAFETY: android-activity initialized ndk-context with a valid VM.
    let vm = unsafe { JavaVM::from_raw(context.vm().cast())? };
    let mut env = vm.attach_current_thread_permanently()?;
    Ok(f(&mut env, activity.as_obj()))
}
