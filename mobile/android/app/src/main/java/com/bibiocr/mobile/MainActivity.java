package com.bibiocr.mobile;

import com.google.androidgamesdk.GameActivity;

public final class MainActivity extends GameActivity {
    static {
        System.loadLibrary("bibiocr_mobile");
    }
}
