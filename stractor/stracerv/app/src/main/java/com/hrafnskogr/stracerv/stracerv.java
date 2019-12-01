package com.hrafnskogr.stracerv;

import android.app.Service;
import android.content.Intent;
import android.os.IBinder;
import android.widget.Toast;
import android.util.Log;

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;

public class stracerv extends Service
{
    private static final String TAG = "stracerv";

    public stracerv() {}

    @Override
    public int onStartCommand(Intent intent, int flags, int startId)
    {
        // Apparently necessary to hook on the UI thread and catch all syscall
        Toast.makeText(getApplicationContext(), "Starting Strace Service", Toast.LENGTH_LONG).show();

        Log.d(TAG, "onStartCommand");
        Log.i(TAG, intent.getData().toString());

        /*final Thread th = new Thread(new Runnable() {
            @Override
            public void run()
            {
                runStrace();
            }
        });

        th.start();*/
        int zygotePid= stracerv.getZygotePid();

        Log.i("info", String.format("Zygote pid %d", zygotePid));
        //Log.i(TAG, "v0.4");

        if(zygotePid != -1)
        {
            String[] cmds = {String.format("strace -p %d -f -tt -T -s 500 -r -o /sdcard/straces.txt", zygotePid)};
            RunAsRoot(cmds);
            /*try
            {
                Process p = Runtime.getRuntime().exec(String.format("strace -p %d -f -tt -T -s 500 -r -o /sdcard/straces.txt", zygotePid));
                Log.i(TAG, "After process");
            }
            catch(IOException e)
            {
                Log.e("CMD error", "IOException "+ e.toString());
            }*/
        }
        return Service.START_NOT_STICKY;
    }

    public void RunAsRoot(String[] cmds)
    {
        try {
            Process p = Runtime.getRuntime().exec("su");
            DataOutputStream os = new DataOutputStream(p.getOutputStream());
            for (String tmpCmd : cmds) {
                os.writeBytes(tmpCmd + "\n");
            }
            os.writeBytes("exit\n");
            os.flush();
        }
        catch(IOException e)
        {
            Log.e("RunAsRoot", "Exception");
        }
    }

    @Override
    public void onDestroy()
    {
        Log.d(TAG, "onDestroy");
    }

    @Override
    public IBinder onBind(Intent intent)
    {
        // TODO: Return the communication channel to the service.
        throw new UnsupportedOperationException("Not yet implemented");
    }

    public static int getZygotePid()
    {
        File[] files = new File("/proc").listFiles();
        for(File file : files)
        {
            if(file.isDirectory())
            {
                int pid;
                try
                {
                    pid = Integer.parseInt(file.getName());

                    final File cmdline = new File("/proc/"+file.getName()+"/cmdline");

                    if(cmdline.exists())
                    {
                        try
                        {
                           FileInputStream fis = new FileInputStream(cmdline);
                           BufferedReader reader = new BufferedReader(new InputStreamReader(fis));
                           String line = reader.readLine();

                           try
                           {
                               if (line.startsWith("zygote")) {
                                   reader.close();
                                   return pid;
                               }
                           }
                           catch(NullPointerException e)
                           {
                                continue;
                           }
                        }
                        catch(IOException e)
                        {
                            Log.e("error", "bug");
                        }
                    }
                }
                catch(NumberFormatException e)
                {
                    continue;
                }
            }
        }
        return -1;
    }
}
