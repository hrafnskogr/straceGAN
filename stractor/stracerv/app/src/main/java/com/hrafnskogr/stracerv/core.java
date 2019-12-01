package com.hrafnskogr.stracerv;

import android.app.Activity;
import android.os.Bundle;
import android.widget.Toast;

public class core extends Activity
{
    public void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);
        //Toast.makeText(getBaseContext(), "Hayaa....", Toast.LENGTH_LONG).show();
    }
}
