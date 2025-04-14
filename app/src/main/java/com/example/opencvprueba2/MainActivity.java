package com.example.opencvprueba2;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

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
import java.io.InputStream;

public class MainActivity extends AppCompatActivity {

    ImageView imageView;
    Button btnLoadImage, btnDetectFaces;
    Bitmap selectedBitmap;
    CascadeClassifier faceDetector;

    static final int PICK_IMAGE = 1;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        imageView = findViewById(R.id.imageView);
        btnLoadImage = findViewById(R.id.btnLoadImage);
        btnDetectFaces = findViewById(R.id.btnDetectFaces);

        // Cargar OpenCV
        if (!OpenCVLoader.initDebug()) {
            Toast.makeText(this, "Error cargando OpenCV", Toast.LENGTH_SHORT).show();
        } else {
            initCascade();
        }

        // Botón para cargar imagen desde galería
        btnLoadImage.setOnClickListener(view -> {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE)
                    != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this,
                        new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, 100);
            } else {
                openGallery();
            }
        });

        // Botón para detectar caras
        btnDetectFaces.setOnClickListener(view -> {
            if (selectedBitmap != null && faceDetector != null) {
                detectFaces(selectedBitmap);
            } else {
                Toast.makeText(this, "Primero carga una imagen", Toast.LENGTH_SHORT).show();
            }
        });
    }

    // Galería
    void openGallery() {
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(intent, PICK_IMAGE);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == PICK_IMAGE && resultCode == Activity.RESULT_OK && data != null) {
            Uri imageUri = data.getData();
            try {
                selectedBitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), imageUri);
                imageView.setImageBitmap(selectedBitmap);
            } catch (Exception e) {
                e.printStackTrace();
                Toast.makeText(this, "Error cargando imagen", Toast.LENGTH_SHORT).show();
            }
        }
    }

    // Inicializar Haar Cascade desde raw/
    void initCascade() {
        try {
            InputStream is = getAssets().open("haarcascade_frontalface_default.xml");
            File cascadeDir = getDir("cascade", MODE_PRIVATE);
            File mCascadeFile = new File(cascadeDir, "haarcascade_frontalface_default.xml");
            FileOutputStream os = new FileOutputStream(mCascadeFile);

            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();

            faceDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());
            if (faceDetector.empty()) {
                faceDetector = null;
                Toast.makeText(this, "No se pudo cargar el detector", Toast.LENGTH_SHORT).show();
            }

        } catch (Exception e) {
            Log.e("OpenCV", "Error cargando cascade", e);
        }
    }

    // Detección de rostros
    void detectFaces(Bitmap bitmap) {
        Mat mat = new Mat();
        Utils.bitmapToMat(bitmap, mat);
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2GRAY);
        Imgproc.equalizeHist(mat, mat);  // mejora contraste

        MatOfRect faces = new MatOfRect();
        faceDetector.detectMultiScale(mat, faces, 1.1, 5,
                0, new Size(100, 100), new Size());

        Rect[] faceArray = faces.toArray();
        if (faceArray.length == 0) {
            Toast.makeText(this, "No se detectaron caras", Toast.LENGTH_SHORT).show();
            return;
        }

        // Dibujar rectángulos
        Utils.bitmapToMat(bitmap, mat);  // convertir original
        for (Rect rect : faceArray) {
            Imgproc.rectangle(mat, rect.tl(), rect.br(), new Scalar(0, 255, 0), 4);
        }

        Bitmap resultBitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(mat, resultBitmap);
        imageView.setImageBitmap(resultBitmap);
    }
}

