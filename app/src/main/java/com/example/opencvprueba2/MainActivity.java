package com.example.opencvprueba2;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;

import androidx.appcompat.app.AppCompatActivity;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class MainActivity extends AppCompatActivity {
    private ImageView imageView;
    private CascadeClassifier faceCascade;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // 1) Inicializar OpenCV en modo debug
        if (!OpenCVLoader.initDebug()) {
            Log.e("OpenCV", "No se pudo inicializar OpenCV");
            return;
        }

        setContentView(R.layout.activity_main);
        imageView = findViewById(R.id.imageView);

        // 2) Cargar la imagen de prueba desde drawable
        Bitmap bmp = BitmapFactory.decodeResource(getResources(), R.drawable.sample_face);
        Mat imgMat = new Mat();
        Utils.bitmapToMat(bmp, imgMat);

        // 3) Convertir a escala de grises
        Mat grayMat = new Mat();
        Imgproc.cvtColor(imgMat, grayMat, Imgproc.COLOR_BGR2GRAY);

        // 4) Cargar el clasificador Haar desde assets
        try {
            InputStream is = getAssets().open("haarcascade_frontalface_default.xml");
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
            File cascadeFile = new File(cascadeDir, "haarcascade_frontalface_default.xml");
            FileOutputStream os = new FileOutputStream(cascadeFile);
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();

            faceCascade = new CascadeClassifier(cascadeFile.getAbsolutePath());
            if (faceCascade.empty()) {
                Log.e("OpenCV", "Error al cargar CascadeClassifier desde assets");
                faceCascade = null;
            }
        } catch (IOException e) {
            Log.e("OpenCV", "Error cargando cascade desde assets", e);
            return;
        }

        // 5) Detectar caras
        MatOfRect faces = new MatOfRect();
        faceCascade.detectMultiScale(
                grayMat,
                faces,
                1.1,     // scaleFactor
                2,       // minNeighbors
                0,
                new Size(100, 100), // minSize
                new Size()          // maxSize
        );

        // 6) Dibujar rectángulos verdes sobre cada cara detectada
        for (Rect r : faces.toArray()) {
            Imgproc.rectangle(
                    imgMat,
                    r.tl(),
                    r.br(),
                    new Scalar(0, 255, 0, 255),
                    4
            );
        }

        // 7) Mostrar resultado en el ImageView
        Bitmap outBmp = Bitmap.createBitmap(imgMat.cols(), imgMat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(imgMat, outBmp);
        imageView.setImageBitmap(outBmp);
    }
}
