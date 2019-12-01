package com.hrafnskogr.stracerv;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.os.Build;
import android.util.Log;
import android.widget.Toast;

public class startStracerv extends BroadcastReceiver
{
    @Override
    public void onReceive(Context context, Intent arg1)
    {
        //Toast.makeText(context, "onReceived", Toast.LENGTH_LONG);

        Intent intent = new Intent(context, stracerv.class);
        if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.O)
        {
            context.startForegroundService(intent);
        }
        else
        {
            context.startService(intent);
        }

    }
}
