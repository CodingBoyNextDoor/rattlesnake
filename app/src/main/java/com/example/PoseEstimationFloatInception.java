/*
 * Copyright 2018 Zihua Zeng (edvard_hua@live.com)
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.example;

import android.app.Activity;
import android.graphics.PointF;
import android.util.Log;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import com.xiaomi.mace.JniMaceUtils;

import java.io.IOException;


public class PoseEstimationFloatInception extends PoseEstimation {
    private Mat mMat;

    /**
     * Initializes an {@code PoseEstimation}.
     *
     * @param activity
     */
    public PoseEstimationFloatInception(Activity activity) throws IOException {
        super(activity);
    }


    @Override
    protected int getImageSizeX() {
        return 192;
    }

    @Override
    protected int getImageSizeY() {
        return 192;
    }

    @Override
    protected int getOutputSizeX() {
        return 96;
    }

    @Override
    protected int getOutputSizeY() {
        return 96;
    }

    @Override
    protected void addPixelValue(int pixelValue) {
        //bgr
        floatBuffer.put(pixelValue & 0xFF);
        floatBuffer.put((pixelValue >> 8) & 0xFF);
        floatBuffer.put((pixelValue >> 16) & 0xFF);
    }


    @Override
    protected void runInference() {
        float[] result = JniMaceUtils.maceMobilenetClassify(floatBuffer.array());

        if (mPrintPointArray == null)
            mPrintPointArray = new float[2][14];


        //先进行高斯滤波,5*5
        if (mMat == null)
            mMat = new Mat(96, 96, CvType.CV_32F);

        float[] tempArray = new float[getOutputSizeY() * getOutputSizeX()];
        float[] outTempArray = new float[getOutputSizeY() * getOutputSizeX()];

        long st = System.currentTimeMillis();

        for (int i = 0; i < 14; i++) {
            int index = 0;
            for (int x = 0; x < 96; x++) {
                for (int y = 0; y < 96; y++) {
                    tempArray[index] = result[x * getOutputSizeY() * 14 + y * 14 + i];
                    index++;
                }
            }

            mMat.put(0, 0, tempArray);
            Imgproc.GaussianBlur(mMat, mMat, new Size(5, 5), 0, 0);
            mMat.get(0, 0, outTempArray);

            float maxX = 0, maxY = 0;
            float max = 0;

            for (int x = 0; x < getOutputSizeX(); x++) {
                for (int y = 0; y < getOutputSizeY(); y++) {
                    float center = get(x, y, outTempArray);

                    if (center >= 0.01) {

                        if (center > max) {
                            max = center;
                            maxX = x;
                            maxY = y;
                        }
                    }
                }
            }

            if (max == 0) {
                mPrintPointArray = new float[2][14];
                return;
            }

            mPrintPointArray[0][i] = maxY;
            mPrintPointArray[1][i] = maxX;
        }

        Log.i("post_processing", "" + (System.currentTimeMillis() - st));
    }

    private float get(int x, int y, float[] arr) {
        if (x < 0 || y < 0 || x >= getOutputSizeX() || y >= getOutputSizeY())
            return -1;
        return arr[x * getOutputSizeX() + y];
    }
    @Override
    public PointF getHead() {
        PointF pointF=new PointF(mPrintPointArray[0][0],mPrintPointArray[1][0]);
        return pointF;
    }

    @Override
    public PointF getNeck() {
        PointF pointF=new PointF(mPrintPointArray[0][1],mPrintPointArray[1][1]);
        return pointF;
    }

    @Override
    public PointF getRShoulder() {
        PointF pointF=new PointF(mPrintPointArray[0][2],mPrintPointArray[1][2]);
        return pointF;
    }

    @Override
    public PointF getRElbow() {
        PointF pointF=new PointF(mPrintPointArray[0][3],mPrintPointArray[1][3]);
        return pointF;
    }

    @Override
    public PointF getRWrist() {
        PointF pointF=new PointF(mPrintPointArray[0][4],mPrintPointArray[1][4]);
        return pointF;
    }

    @Override
    public PointF getLShoulder() {
        PointF pointF=new PointF(mPrintPointArray[0][5],mPrintPointArray[1][5]);
        return pointF;
    }

    @Override
    public PointF getLElbow() {
        PointF pointF=new PointF(mPrintPointArray[0][6],mPrintPointArray[1][6]);
        return pointF;
    }

    @Override
    public PointF getLWrist() {
        PointF pointF=new PointF(mPrintPointArray[0][7],mPrintPointArray[1][7]);
        return pointF;
    }

    @Override
    public PointF getRHip() {
        PointF pointF=new PointF(mPrintPointArray[0][8],mPrintPointArray[1][8]);
        return pointF;
    }

    @Override
    public PointF getRKnee() {
        PointF pointF=new PointF(mPrintPointArray[0][9],mPrintPointArray[1][9]);
        return pointF;
    }

    @Override
    public PointF getRAnkle() {
        PointF pointF=new PointF(mPrintPointArray[0][10],mPrintPointArray[1][10]);
        return pointF;
    }

    @Override
    public PointF getLHip() {
        PointF pointF=new PointF(mPrintPointArray[0][11],mPrintPointArray[1][11]);
        return pointF;
    }

    @Override
    public PointF getLKnee() {
        PointF pointF=new PointF(mPrintPointArray[0][12],mPrintPointArray[1][12]);
        return pointF;
    }

    @Override
    public PointF getLankle() {
        PointF pointF=new PointF(mPrintPointArray[0][13],mPrintPointArray[1][13]);
        return pointF;
    }
}
