package com.bibiocr.mobile;

import android.Manifest;
import android.app.Activity;
import android.app.AlertDialog;
import android.app.Fragment;
import android.content.ClipData;
import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.provider.MediaStore;
import android.text.InputType;
import android.view.WindowManager;
import android.widget.EditText;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.concurrent.atomic.AtomicBoolean;

public final class CameraFragment extends Fragment {
    private static final int CAMERA = 0x4249;
    private static final int PERMISSION = 0x424a;
    private static final String TAG = "com.bibiocr.mobile.CameraFragment";
    private long callback;
    private Uri output;

    private static native void rustCallback(long callback, String path, String error);
    private static native void rustTextCallback(long callback, String text, String error);

    public static void showText(Activity activity, long callback, String title, String initial, boolean multiline) {
        activity.runOnUiThread(() -> {
            AtomicBoolean delivered = new AtomicBoolean();
            try {
                EditText editor = new EditText(activity);
                editor.setInputType(multiline
                        ? InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_MULTI_LINE
                        : InputType.TYPE_CLASS_TEXT);
                if (multiline) editor.setMinLines(6);
                editor.setText(initial);
                editor.setSelection(editor.length());
                AlertDialog dialog = new AlertDialog.Builder(activity)
                        .setTitle(title)
                        .setView(editor)
                        .setPositiveButton(android.R.string.ok, (ignored, which) -> {
                            if (delivered.compareAndSet(false, true))
                                rustTextCallback(callback, editor.getText().toString(), null);
                        })
                        .setNegativeButton(android.R.string.cancel, (ignored, which) -> {
                            if (delivered.compareAndSet(false, true)) rustTextCallback(callback, null, null);
                        })
                        .create();
                dialog.setOnDismissListener(ignored -> {
                    if (delivered.compareAndSet(false, true)) rustTextCallback(callback, null, null);
                });
                dialog.show();
                dialog.getWindow().setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_STATE_ALWAYS_VISIBLE);
            } catch (Throwable error) {
                if (delivered.compareAndSet(false, true)) rustTextCallback(callback, null, error.toString());
            }
        });
    }

    public CameraFragment() {}
    private CameraFragment(long callback) { this.callback = callback; }

    public static boolean show(Activity activity, long callback) {
        if (activity.getFragmentManager().findFragmentByTag(TAG) != null) return false;
        activity.runOnUiThread(() -> activity.getFragmentManager().beginTransaction()
                .add(new CameraFragment(callback), TAG).commitAllowingStateLoss());
        return true;
    }

    @Override public void onResume() {
        super.onResume();
        if (callback == 0 || output != null) return;
        boolean cameraDenied = getActivity().checkSelfPermission(Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED;
        boolean storageDenied = Build.VERSION.SDK_INT < 29
                && getActivity().checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED;
        if (cameraDenied || storageDenied) {
            requestPermissions(
                    Build.VERSION.SDK_INT < 29
                            ? new String[]{Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE}
                            : new String[]{Manifest.permission.CAMERA},
                    PERMISSION);
        } else {
            launch();
        }
    }

    @Override public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] results) {
        if (requestCode != PERMISSION) return;
        for (int result : results) {
            if (result != PackageManager.PERMISSION_GRANTED) {
                finish(null, "Camera or storage permission denied");
                return;
            }
        }
        launch();
    }

    private void launch() {
        try {
            ContentValues values = new ContentValues();
            values.put(MediaStore.Images.Media.DISPLAY_NAME, "bibiocr-" + System.currentTimeMillis() + ".jpg");
            values.put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg");
            if (Build.VERSION.SDK_INT >= 29) {
                values.put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/BIBIOCR");
            }
            output = getActivity().getContentResolver().insert(
                    MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
            if (output == null) throw new IllegalStateException("Cannot create camera output");
            Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
            intent.putExtra(MediaStore.EXTRA_OUTPUT, output);
            intent.setClipData(ClipData.newRawUri("bibiocr-photo", output));
            intent.addFlags(Intent.FLAG_GRANT_WRITE_URI_PERMISSION | Intent.FLAG_GRANT_READ_URI_PERMISSION);
            startActivityForResult(intent, CAMERA);
        } catch (Throwable error) {
            finish(null, error.toString());
        }
    }

    @Override public void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode != CAMERA) return;
        Activity activity = getActivity();
        Uri captured = output;
        if (activity == null || captured == null) {
            finish(null, "Camera did not return an output image");
            return;
        }
        new Thread(() -> {
            String path = null;
            String failure = null;
            try {
                ContentResolver resolver = activity.getContentResolver();
                // A camera may return before its output stream closes, and some
                // vendors report CANCELED despite having written EXTRA_OUTPUT.
                boolean hasImage = false;
                for (int attempt = 0; attempt < 10 && !hasImage; attempt++) {
                    try (InputStream probe = resolver.openInputStream(captured)) {
                        hasImage = probe != null && probe.read() >= 0;
                    } catch (FileNotFoundException missing) {
                        // The media row exists but the camera has not created its file yet.
                    }
                    if (!hasImage) Thread.sleep(100);
                }
                if (!hasImage) {
                    resolver.delete(captured, null, null);
                    String message = resultCode == Activity.RESULT_OK
                            ? "Camera did not save an image" : null;
                    activity.runOnUiThread(() -> finish(null, message));
                    return;
                }
                File dir = new File(activity.getFilesDir(), "scans");
                if (!dir.exists() && !dir.mkdirs()) throw new IllegalStateException("Cannot create scans directory");
                File file = new File(dir, "scan-" + System.currentTimeMillis() + ".jpg");
                try (InputStream in = resolver.openInputStream(captured);
                     FileOutputStream out = new FileOutputStream(file)) {
                    if (in == null) throw new IllegalStateException("Cannot read captured image");
                    byte[] buffer = new byte[262144];
                    int count;
                    while ((count = in.read(buffer)) >= 0) out.write(buffer, 0, count);
                }
                path = file.getAbsolutePath();
            } catch (Throwable error) {
                failure = error.toString();
            }
            String completedPath = path;
            String completedFailure = failure;
            activity.runOnUiThread(() -> finish(completedPath, completedFailure));
        }, "bibiocr-photo-copy").start();
    }

    private void finish(String path, String error) {
        long value = callback;
        callback = 0;
        if (value != 0) rustCallback(value, path, error);
        if (getFragmentManager() != null) getFragmentManager().beginTransaction().remove(this).commitAllowingStateLoss();
    }
}
