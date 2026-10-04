use jni::{
    JNIEnv, NativeMethod,
    objects::{GlobalRef, JClass, JObject, JString, JValueGen},
    sys::jlong,
};
use std::{
    path::PathBuf,
    sync::{
        Arc, OnceLock,
        atomic::{AtomicBool, AtomicU32, Ordering},
    },
    time::{Duration, Instant},
};

type CameraCallback = Box<dyn FnOnce(Result<Option<PathBuf>, String>) + Send + 'static>;
type TextCallback = Box<dyn FnOnce(Result<Option<String>, String>) + Send + 'static>;

const CAMERA_DEX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/classes.dex"));
static CAMERA_CLASS: OnceLock<GlobalRef> = OnceLock::new();

pub struct OcrWakeLock(GlobalRef);

impl OcrWakeLock {
    pub fn acquire() -> Result<Self, String> {
        robius_android_env::with_activity(|env, activity| -> jni::errors::Result<Self> {
            let service = env.new_string("power")?;
            let manager = env
                .call_method(
                    activity,
                    "getSystemService",
                    "(Ljava/lang/String;)Ljava/lang/Object;",
                    &[JValueGen::Object(service.as_ref())],
                )?
                .l()?;
            let tag = env.new_string("com.bibiocr.mobile:ocr")?;
            let lock = env
                .call_method(
                    &manager,
                    "newWakeLock",
                    "(ILjava/lang/String;)Landroid/os/PowerManager$WakeLock;",
                    &[JValueGen::Int(1), JValueGen::Object(tag.as_ref())],
                )?
                .l()?;
            env.call_method(&lock, "acquire", "(J)V", &[JValueGen::Long(30 * 60 * 1000)])?;
            env.new_global_ref(lock).map(Self)
        })
        .map_err(|error| error.to_string())?
        .map_err(|error| error.to_string())
    }
}

impl Drop for OcrWakeLock {
    fn drop(&mut self) {
        let _ = robius_android_env::with_activity(|env, _| {
            let _ = env.call_method(self.0.as_obj(), "release", "()V", &[]);
        });
    }
}

pub fn system_insets_px() -> Result<Option<crate::core::SystemInsetsPx>, String> {
    robius_android_env::with_activity(|env, activity| {
        let result = env.with_local_frame(32, |env| read_system_insets(env, activity));
        if env.exception_check().unwrap_or(false) {
            let _ = env.exception_clear();
        }
        result.map_err(|error| error.to_string())
    })
    .map_err(|error| error.to_string())?
}

pub fn is_wifi_connected() -> Result<bool, String> {
    robius_android_env::with_activity(|env, activity| {
        env.with_local_frame(32, |env| read_wifi_connected(env, activity))
            .map_err(|error| error.to_string())
    })
    .map_err(|error| error.to_string())?
}

fn read_wifi_connected(env: &mut JNIEnv<'_>, activity: &JObject<'_>) -> jni::errors::Result<bool> {
    let name = JObject::from(env.new_string("connectivity")?);
    let manager = env
        .call_method(
            activity,
            "getSystemService",
            "(Ljava/lang/String;)Ljava/lang/Object;",
            &[JValueGen::Object(&name)],
        )?
        .l()?;
    if manager.is_null() {
        return Ok(false);
    }
    let network = env
        .call_method(&manager, "getActiveNetwork", "()Landroid/net/Network;", &[])?
        .l()?;
    if network.is_null() {
        return Ok(false);
    }
    let capabilities = env
        .call_method(
            &manager,
            "getNetworkCapabilities",
            "(Landroid/net/Network;)Landroid/net/NetworkCapabilities;",
            &[JValueGen::Object(&network)],
        )?
        .l()?;
    if capabilities.is_null() {
        return Ok(false);
    }
    env.call_method(&capabilities, "hasTransport", "(I)Z", &[JValueGen::Int(1)])?
        .z()
}

fn read_system_insets(
    env: &mut JNIEnv<'_>,
    activity: &JObject<'_>,
) -> jni::errors::Result<Option<crate::core::SystemInsetsPx>> {
    let window = env
        .call_method(activity, "getWindow", "()Landroid/view/Window;", &[])?
        .l()?;
    let decor = env
        .call_method(&window, "getDecorView", "()Landroid/view/View;", &[])?
        .l()?;
    let insets = env
        .call_method(
            &decor,
            "getRootWindowInsets",
            "()Landroid/view/WindowInsets;",
            &[],
        )?
        .l()?;

    let mut top = 0;
    let mut bottom = 0;
    if !insets.is_null() {
        top = env
            .call_method(&insets, "getSystemWindowInsetTop", "()I", &[])?
            .i()?
            .max(
                env.call_method(&insets, "getStableInsetTop", "()I", &[])?
                    .i()?,
            );
        bottom = env
            .call_method(&insets, "getSystemWindowInsetBottom", "()I", &[])?
            .i()?
            .max(
                env.call_method(&insets, "getStableInsetBottom", "()I", &[])?
                    .i()?,
            );
        let sdk = env
            .get_static_field("android/os/Build$VERSION", "SDK_INT", "I")?
            .i()?;
        if sdk >= 28 {
            let cutout = env
                .call_method(
                    &insets,
                    "getDisplayCutout",
                    "()Landroid/view/DisplayCutout;",
                    &[],
                )?
                .l()?;
            if !cutout.is_null() {
                top = top.max(
                    env.call_method(&cutout, "getSafeInsetTop", "()I", &[])?
                        .i()?,
                );
                bottom = bottom.max(
                    env.call_method(&cutout, "getSafeInsetBottom", "()I", &[])?
                        .i()?,
                );
            }
        }
    }
    if top == 0 {
        top = system_dimen_px(env, activity, "status_bar_height")?;
    }
    if bottom == 0 {
        bottom = system_dimen_px(env, activity, "navigation_bar_height")?;
    }
    Ok(Some(crate::core::SystemInsetsPx {
        top: top.max(0) as f32,
        bottom: bottom.max(0) as f32,
    }))
}

fn system_dimen_px(
    env: &mut JNIEnv<'_>,
    activity: &JObject<'_>,
    name: &str,
) -> jni::errors::Result<i32> {
    let resources = env
        .call_method(
            activity,
            "getResources",
            "()Landroid/content/res/Resources;",
            &[],
        )?
        .l()?;
    let name = env.new_string(name)?;
    let kind = env.new_string("dimen")?;
    let package = env.new_string("android")?;
    let id = env
        .call_method(
            &resources,
            "getIdentifier",
            "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)I",
            &[
                JValueGen::Object(name.as_ref()),
                JValueGen::Object(kind.as_ref()),
                JValueGen::Object(package.as_ref()),
            ],
        )?
        .i()?;
    if id == 0 {
        return Ok(0);
    }
    env.call_method(
        &resources,
        "getDimensionPixelSize",
        "(I)I",
        &[JValueGen::Int(id)],
    )?
    .i()
}

pub fn capture_photo<F>(callback: F) -> Result<(), String>
where
    F: FnOnce(Result<Option<PathBuf>, String>) + Send + 'static,
{
    let callback: CameraCallback = Box::new(callback);
    let callback_ptr = Box::into_raw(Box::new(callback));
    let result = robius_android_env::with_activity(|env, activity| {
        let class = camera_class(env).map_err(|error| error.to_string())?;
        let shown = env
            .call_static_method(
                class,
                "show",
                "(Landroid/app/Activity;J)Z",
                &[
                    JValueGen::Object(activity),
                    JValueGen::Long(callback_ptr as i64),
                ],
            )
            .and_then(|value| value.z())
            .map_err(|error| error.to_string())?;
        shown
            .then_some(())
            .ok_or_else(|| "Camera is already open".to_owned())
    });
    match result {
        Ok(inner) => {
            if inner.is_err() {
                // SAFETY: Java never received ownership when `show` failed.
                let _ = unsafe { Box::from_raw(callback_ptr) };
            }
            inner
        }
        Err(error) => {
            // SAFETY: no Activity was available, so Java cannot own this pointer.
            let _ = unsafe { Box::from_raw(callback_ptr) };
            Err(error.to_string())
        }
    }
}

pub fn prompt_text<F>(
    title: &str,
    initial: &str,
    multiline: bool,
    callback: F,
) -> Result<(), String>
where
    F: FnOnce(Result<Option<String>, String>) + Send + 'static,
{
    let callback: TextCallback = Box::new(callback);
    let pointer = Box::into_raw(Box::new(callback));
    let result = robius_android_env::with_activity(|env, activity| -> jni::errors::Result<()> {
        let class = camera_class(env)?;
        let title = env.new_string(title)?;
        let initial = env.new_string(initial)?;
        env.call_static_method(
            class,
            "showText",
            "(Landroid/app/Activity;JLjava/lang/String;Ljava/lang/String;Z)V",
            &[
                JValueGen::Object(activity),
                JValueGen::Long(pointer as i64),
                JValueGen::Object(title.as_ref()),
                JValueGen::Object(initial.as_ref()),
                JValueGen::Bool(u8::from(multiline)),
            ],
        )?;
        Ok(())
    })
    .map_err(|error| error.to_string())
    .and_then(|inner| inner.map_err(|error| error.to_string()));
    if result.is_err() {
        // SAFETY: Java never received ownership if invocation failed.
        let _ = unsafe { Box::from_raw(pointer) };
    }
    result
}

pub fn share_text(text: &str) -> Result<(), String> {
    share("text/markdown", Some(text), None)
}

fn write_audio_track(
    env: &mut JNIEnv<'_>,
    track: &JObject<'_>,
    pcm: &[i16],
    stop: &AtomicBool,
) -> Result<usize, String> {
    if pcm.is_empty() {
        return Ok(0);
    }
    let array = env
        .new_short_array(pcm.len() as i32)
        .map_err(|error| error.to_string())?;
    env.set_short_array_region(&array, 0, pcm)
        .map_err(|error| error.to_string())?;
    let mut offset = 0;
    while offset < pcm.len() && !stop.load(Ordering::Relaxed) {
        let written = env
            .call_method(
                track,
                "write",
                "([SII)I",
                &[
                    JValueGen::Object(array.as_ref()),
                    JValueGen::Int(offset as i32),
                    JValueGen::Int((pcm.len() - offset) as i32),
                ],
            )
            .and_then(|value| value.i())
            .map_err(|error| error.to_string())?;
        if written <= 0 {
            return Err(format!("AudioTrack write failed: {written}"));
        }
        offset += written as usize;
    }
    env.delete_local_ref(array)
        .map_err(|error| error.to_string())?;
    Ok(offset)
}

pub fn play_pcm(
    samples: &[i16],
    sample_rate: i32,
    volume: &Arc<AtomicU32>,
    speed: &Arc<AtomicU32>,
    stop: &Arc<AtomicBool>,
) -> Result<(), String> {
    if sample_rate <= 0 {
        return Err("Invalid TTS sample rate".to_owned());
    }
    let mut stretcher =
        wsola::TimeStretch::new(sample_rate as u32, 1).map_err(|error| error.to_string())?;
    robius_android_env::with_activity(|env, _| {
        let result = (|| {
            let min_buffer = env
                .call_static_method(
                    "android/media/AudioTrack",
                    "getMinBufferSize",
                    "(III)I",
                    &[
                        JValueGen::Int(sample_rate),
                        JValueGen::Int(4),
                        JValueGen::Int(2),
                    ],
                )
                .and_then(|value| value.i())
                .map_err(|error| error.to_string())?;
            if min_buffer <= 0 {
                return Err(format!("AudioTrack buffer error: {min_buffer}"));
            }
            let track = env
                .new_object(
                    "android/media/AudioTrack",
                    "(IIIIII)V",
                    &[
                        JValueGen::Int(3),
                        JValueGen::Int(sample_rate),
                        JValueGen::Int(4),
                        JValueGen::Int(2),
                        JValueGen::Int(min_buffer.max(4096)),
                        JValueGen::Int(1),
                    ],
                )
                .map_err(|error| error.to_string())?;
            let result = (|| {
                env.call_method(&track, "play", "()V", &[])
                    .map_err(|error| error.to_string())?;
                let mut queued = 0;
                for chunk in samples.chunks(2048) {
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }
                    let rate = f32::from_bits(speed.load(Ordering::Relaxed));
                    if ![0.5, 0.75, 1.0, 1.25, 1.5, 2.0].contains(&rate) {
                        return Err("Unsupported playback speed".to_owned());
                    }
                    let pcm = crate::tts::stretch_pcm_chunk(&mut stretcher, chunk, rate, false);
                    let level = f32::from_bits(volume.load(Ordering::Relaxed)).clamp(0.0, 1.0);
                    env.call_method(&track, "setVolume", "(F)I", &[JValueGen::Float(level)])
                        .map_err(|error| error.to_string())?;
                    queued += write_audio_track(env, &track, &pcm, stop)?;
                }
                if !stop.load(Ordering::Relaxed) {
                    let rate = f32::from_bits(speed.load(Ordering::Relaxed));
                    let tail = crate::tts::stretch_pcm_chunk(&mut stretcher, &[], rate, true);
                    queued += write_audio_track(env, &track, &tail, stop)?;
                }
                // AudioTrack.write only queues PCM; stop() would discard its unplayed tail.
                let deadline = Instant::now()
                    + Duration::from_secs_f64(
                        samples.len() as f64 / sample_rate as f64 / 0.5 + 5.0,
                    );
                while !stop.load(Ordering::Relaxed) {
                    let played =
                        env.call_method(&track, "getPlaybackHeadPosition", "()I", &[])
                            .and_then(|value| value.i())
                            .map_err(|error| error.to_string())? as u32;
                    if played as usize >= queued {
                        break;
                    }
                    if Instant::now() >= deadline {
                        return Err("AudioTrack did not finish playback".to_owned());
                    }
                    std::thread::sleep(Duration::from_millis(20));
                }
                Ok(())
            })();
            // Clear any pending Java exception before stop/release or thread detach.
            if env.exception_check().unwrap_or(false) {
                let _ = env.exception_clear();
            }
            let _ = env.call_method(&track, "stop", "()V", &[]);
            let _ = env.call_method(&track, "release", "()V", &[]);
            if env.exception_check().unwrap_or(false) {
                let _ = env.exception_clear();
            }
            let _ = env.delete_local_ref(track);
            result
        })();
        if env.exception_check().unwrap_or(false) {
            let _ = env.exception_clear();
        }
        result
    })
    .map_err(|error| error.to_string())?
}

pub fn share_uri(uri: &str, mime: &str) -> Result<(), String> {
    share(mime, None, Some(uri))
}

fn share(mime: &str, text: Option<&str>, uri: Option<&str>) -> Result<(), String> {
    robius_android_env::with_activity(|env, activity| {
        let intent_class = env
            .find_class("android/content/Intent")
            .map_err(|error| error.to_string())?;
        let action = env
            .new_string("android.intent.action.SEND")
            .map_err(|error| error.to_string())?;
        let intent = env
            .new_object(
                &intent_class,
                "(Ljava/lang/String;)V",
                &[JValueGen::Object(action.as_ref())],
            )
            .map_err(|error| error.to_string())?;
        let mime = env.new_string(mime).map_err(|error| error.to_string())?;
        env.call_method(
            &intent,
            "setType",
            "(Ljava/lang/String;)Landroid/content/Intent;",
            &[JValueGen::Object(mime.as_ref())],
        )
        .map_err(|error| error.to_string())?;
        if let Some(text) = text {
            let key = env
                .new_string("android.intent.extra.TEXT")
                .map_err(|error| error.to_string())?;
            let value = env.new_string(text).map_err(|error| error.to_string())?;
            env.call_method(
                &intent,
                "putExtra",
                "(Ljava/lang/String;Ljava/lang/String;)Landroid/content/Intent;",
                &[
                    JValueGen::Object(key.as_ref()),
                    JValueGen::Object(value.as_ref()),
                ],
            )
            .map_err(|error| error.to_string())?;
        }
        if let Some(uri) = uri {
            let uri_class = env
                .find_class("android/net/Uri")
                .map_err(|error| error.to_string())?;
            let uri = env.new_string(uri).map_err(|error| error.to_string())?;
            let uri = env
                .call_static_method(
                    uri_class,
                    "parse",
                    "(Ljava/lang/String;)Landroid/net/Uri;",
                    &[JValueGen::Object(uri.as_ref())],
                )
                .and_then(|value| value.l())
                .map_err(|error| error.to_string())?;
            let key = env
                .new_string("android.intent.extra.STREAM")
                .map_err(|error| error.to_string())?;
            env.call_method(
                &intent,
                "putExtra",
                "(Ljava/lang/String;Landroid/os/Parcelable;)Landroid/content/Intent;",
                &[JValueGen::Object(key.as_ref()), JValueGen::Object(&uri)],
            )
            .map_err(|error| error.to_string())?;
            env.call_method(
                &intent,
                "addFlags",
                "(I)Landroid/content/Intent;",
                &[JValueGen::Int(1)],
            )
            .map_err(|error| error.to_string())?;
        }
        let chooser = env
            .call_static_method(
                intent_class,
                "createChooser",
                "(Landroid/content/Intent;Ljava/lang/CharSequence;)Landroid/content/Intent;",
                &[
                    JValueGen::Object(&intent),
                    JValueGen::Object(&JObject::null()),
                ],
            )
            .and_then(|value| value.l())
            .map_err(|error| error.to_string())?;
        env.call_method(
            activity,
            "startActivity",
            "(Landroid/content/Intent;)V",
            &[JValueGen::Object(&chooser)],
        )
        .map_err(|error| error.to_string())?;
        Ok(())
    })
    .map_err(|error| error.to_string())?
}

unsafe extern "C" fn camera_callback<'a>(
    mut env: JNIEnv<'a>,
    _: JObject<'a>,
    callback_ptr: jlong,
    path: JString<'a>,
    error: JString<'a>,
) {
    let callback_ptr = callback_ptr as *mut CameraCallback;
    if callback_ptr.is_null() {
        return;
    }
    // SAFETY: the Java fragment claims this pointer exactly once.
    let callback = *unsafe { Box::from_raw(callback_ptr) };
    let result = if !error.as_raw().is_null() {
        env.get_string(&error)
            .map(String::from)
            .map_err(|jni_error| jni_error.to_string())
            .and_then(Err)
    } else if path.as_raw().is_null() {
        Ok(None)
    } else {
        env.get_string(&path)
            .map(|value| Some(PathBuf::from(String::from(value))))
            .map_err(|jni_error| jni_error.to_string())
    };
    std::thread::spawn(move || callback(result));
}

unsafe extern "C" fn text_callback<'a>(
    mut env: JNIEnv<'a>,
    _: JObject<'a>,
    callback_ptr: jlong,
    text: JString<'a>,
    error: JString<'a>,
) {
    let pointer = callback_ptr as *mut TextCallback;
    if pointer.is_null() {
        return;
    }
    // SAFETY: Java's delivered flag claims this pointer exactly once.
    let callback = *unsafe { Box::from_raw(pointer) };
    let result = if !error.as_raw().is_null() {
        env.get_string(&error)
            .map(String::from)
            .map_err(|jni_error| jni_error.to_string())
            .and_then(Err)
    } else if text.as_raw().is_null() {
        Ok(None)
    } else {
        env.get_string(&text)
            .map(|value| Some(String::from(value)))
            .map_err(|jni_error| jni_error.to_string())
    };
    callback(result);
}

fn camera_class<'a>(env: &mut JNIEnv<'a>) -> jni::errors::Result<&'static GlobalRef> {
    if let Some(class) = CAMERA_CLASS.get() {
        return Ok(class);
    }
    let buffer =
        unsafe { env.new_direct_byte_buffer(CAMERA_DEX.as_ptr().cast_mut(), CAMERA_DEX.len())? };
    let loader = env.new_object(
        "dalvik/system/InMemoryDexClassLoader",
        "(Ljava/nio/ByteBuffer;Ljava/lang/ClassLoader;)V",
        &[
            JValueGen::Object(&JObject::from(buffer)),
            JValueGen::Object(&JObject::null()),
        ],
    )?;
    let name = env.new_string("com.bibiocr.mobile.CameraFragment")?;
    let class: JClass<'a> = env
        .call_method(
            loader,
            "loadClass",
            "(Ljava/lang/String;)Ljava/lang/Class;",
            &[JValueGen::Object(name.as_ref())],
        )?
        .l()?
        .into();
    env.register_native_methods(
        &class,
        &[
            NativeMethod {
                name: "rustCallback".into(),
                sig: "(JLjava/lang/String;Ljava/lang/String;)V".into(),
                fn_ptr: camera_callback as *mut _,
            },
            NativeMethod {
                name: "rustTextCallback".into(),
                sig: "(JLjava/lang/String;Ljava/lang/String;)V".into(),
                fn_ptr: text_callback as *mut _,
            },
        ],
    )?;
    let global = env.new_global_ref(class)?;
    Ok(CAMERA_CLASS.get_or_init(|| global))
}
