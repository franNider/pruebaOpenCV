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
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.core.Core;
import org.opencv.core.Point;
import org.opencv.core.MatOfByte;
import org.opencv.core.CvType;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;

public class MainActivity extends AppCompatActivity {

    ImageView imageView;
    Button btnLoadImage, btnDetectFaces;
    Bitmap selectedBitmap;
    Net faceNet;

    static final int PICK_IMAGE = 1;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        imageView = findViewById(R.id.imageView);
        btnLoadImage = findViewById(R.id.btnLoadImage);
        btnDetectFaces = findViewById(R.id.btnDetectFaces);

        if (!OpenCVLoader.initDebug()) {
            Toast.makeText(this, "Error cargando OpenCV", Toast.LENGTH_SHORT).show();
        } else {
            loadDnnModel();
        }

        btnLoadImage.setOnClickListener(view -> {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE)
                    != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this,
                        new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, 100);
            } else {
                openGallery();
            }
        });

        btnDetectFaces.setOnClickListener(view -> {
            if (selectedBitmap != null && faceNet != null) {
                detectFacesDnn(selectedBitmap);
            } else {
                Toast.makeText(this, "Primero carga una imagen", Toast.LENGTH_SHORT).show();
            }
        });
    }

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

                // Escalar si es muy grande
                int maxWidth = 1000;
                if (selectedBitmap.getWidth() > maxWidth) {
                    float ratio = (float) selectedBitmap.getHeight() / selectedBitmap.getWidth();
                    selectedBitmap = Bitmap.createScaledBitmap(selectedBitmap, maxWidth, (int) (maxWidth * ratio), true);
                }

                imageView.setImageBitmap(selectedBitmap);
            } catch (Exception e) {
                e.printStackTrace();
                Toast.makeText(this, "Error cargando imagen", Toast.LENGTH_SHORT).show();
            }
        }
    }

    void loadDnnModel() {
        try {
            // Copiar prototxt
            InputStream isProto = getAssets().open("opencv_face_detector.prototxt");
            File protoFile = new File(getCacheDir(), "opencv_face_detector.prototxt");
            FileOutputStream osProto = new FileOutputStream(protoFile);
            copyStream(isProto, osProto);

            // Copiar caffemodel
            InputStream isModel = getAssets().open("opencv_face_detector.caffemodel");
            File modelFile = new File(getCacheDir(), "opencv_face_detector.caffemodel");
            FileOutputStream osModel = new FileOutputStream(modelFile);
            copyStream(isModel, osModel);

            faceNet = Dnn.readNetFromCaffe(protoFile.getAbsolutePath(), modelFile.getAbsolutePath());
            Log.i("OpenCV", "Modelo DNN cargado exitosamente");

        } catch (Exception e) {
            e.printStackTrace();
            Toast.makeText(this, "Error cargando modelo DNN", Toast.LENGTH_SHORT).show();
        }
    }

    void copyStream(InputStream is, FileOutputStream os) throws Exception {
        byte[] buffer = new byte[4096];
        int bytesRead;
        while ((bytesRead = is.read(buffer)) != -1) {
            os.write(buffer, 0, bytesRead);
        }
        is.close();
        os.close();
    }

    void detectFacesDnn(Bitmap bitmap) {
        Mat img = new Mat();
        Utils.bitmapToMat(bitmap, img);
        Imgproc.cvtColor(img, img, Imgproc.COLOR_RGBA2RGB);

        // Crear blob
        Mat blob = Dnn.blobFromImage(img, 1.0, new Size(300, 300),
                new Scalar(104.0, 177.0, 123.0), false, false);

        faceNet.setInput(blob);
        Mat detections = faceNet.forward();

        int cols = img.cols();
        int rows = img.rows();
        detections = detections.reshape(1, (int) detections.total() / 7);

        int detectionsCount = 0;

        for (int i = 0; i < detections.rows(); i++) {
            double confidence = detections.get(i, 2)[0];
            if (confidence > 0.5) {
                int x1 = (int) (detections.get(i, 3)[0] * cols);
                int y1 = (int) (detections.get(i, 4)[0] * rows);
                int x2 = (int) (detections.get(i, 5)[0] * cols);
                int y2 = (int) (detections.get(i, 6)[0] * rows);

                Imgproc.rectangle(img, new Point(x1, y1), new Point(x2, y2),
                        new Scalar(0, 255, 0), 3);
                detectionsCount++;
            }
        }

        if (detectionsCount == 0) {
            Toast.makeText(this, "No se detectaron rostros", Toast.LENGTH_SHORT).show();
        }

        Bitmap result = Bitmap.createBitmap(img.cols(), img.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(img, result);
        imageView.setImageBitmap(result);
    }
}
