package com.smartfocus.ai

import android.app.Application
import android.util.Log

class SmartFocusApplication : Application() {

    override fun onCreate() {
        super.onCreate()
        Thread.setDefaultUncaughtExceptionHandler { thread, e ->
            Log.e("SmartFocusApp", "Uncaught exception in thread: ${thread.name}", e)
        }
    }
}
