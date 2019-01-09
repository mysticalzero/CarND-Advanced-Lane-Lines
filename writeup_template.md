**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[hls_image]: ./output_images/hls_test4.jpg "HLS Image"
[image1]: ./output_images/undistort/undistort_calibration4.jpg "Undistorted"
[image2]: ./output_images/undistort/test5.jpg "Road Transformed"
[image3]: ./output_images/thresholded/thresholded_test3.jpg "Binary Example"
[image4]: ./output_images/warped/warped_test3.jpg "Warp Example"
[image5]: ./output_images/fitted/line_pixel_test4.jpg "Fit Visual"
[image6]: ./output_images/final/test4.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for performing camera calibration can be found in the section **Camera Calibration** of the Jupyter notebook `code_project.ipynb`.

The code first defines the location of the embedded chessboard corners (i.e. the inner corners where th vertices of four squares meet) in object space -- the object points `obj_p`. The location is given as (x, y, z), with x increasing towards the right, y increasing towards the bottom, z being the elevation from the chessboard plane. As the chessboard is assumed to be planar or flat, the z-component of the coordinates of all corners have the same value of 0. Accordingly, all calibration images share the same object points.

For each calibration image, the location of the corners in image space (i.e. image points) is found using OpenCV's `findChessboardCorners()`. Once all calibration images have been processed in this way, we have a correspondence between the object points and the image points. Together with OpenCV's `calibrateCamera()`, this correspondence is used to find the camera matrix and distortion coefficients.

An example of a distortion-corrected calibration image is shown here. Other examples can be found in `./output_images/undistort/`.
![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Here is another distortion-corrected image using the camera matrix and distortion coefficients obtained previously. Other results can be found in `./output_images/undistort/`.
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

To obtain the thresholded binary image, the undistorted image is first converted to HLS format. Then, gradient-thresholding along the x direction is performed on the S-channel image to obtain the thresholded binary image.

H-channel and L-channel were not considered as lane-related edges are not as prominent as those of the S-channel image (see below). As such, gradients on the H-channel and L-channel are not used.

![alt text][hls_image]

The functions and code used for this task can be found in the section **Binary Image** section of the Jupyter notebook `code_project.ipynb` (see `get_binary_image()` and `sobel_x_thres()`)

An example of the thresholded binary image is shown here. Other examples can be found in `./output_images/thresholded/`.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for performing perspective transform can be found in the **Perspective Transform** section of the Jupyter notebook `code_project.ipynb`.

Before perspective transform can be done, the source points and the destination points for the transform have to be predefined. As such, visual inspection on a test image is first performed to ascertain the pixel coordinates of the source points and the destination points. The mapping between the source and destination pixel coordinates are hardcoded as follows:

```python
src_pts = np.float32([[234, 670],[574, 460],[705, 460],[1050, 670]])
dst_pts = np.float32([[290, straight_line_img.shape[0]],[290, 0],[990, 0],[990, straight_line_img.shape[0]]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 234, 670      | 290, 720      | 
| 574, 560      | 290, 0        |
| 705, 460      | 990, 0        |
| 1050, 670     | 990, 720      |

With these, the perspective transform matrix is found using OpenCV's `getPerspectiveTransform()`.

The actual warping code is contained in `warper()`. It simply takes the perspecive transform matrix and calls OpenCV's `warpPerspective()`.

An example is shown here. Other examples can be found in the folder `./output_images/warped/`.
![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code for this is shown in the **Identification of Lane Line Pixels and the Fitting of Polynomial Lines** section of the Jupyter notebook `code_project.ipynb` (see `find_window_centroids()`, `manual_pixel_search()`, `find_lane_pixels_convolution_standalone()`, and `process_image_standalone()`).

The identification and lane pixels and the fitting of polynomial lines are achieved with the following steps:

1. The thresholded binary image is warped and fed into a procedure to find window centroids (see `find_window_centroids()`).
2. Pixels within the boundaries of windows defined by the identified centroids are extracted by performing convolution on the histogram (see `manual_pixel_search()`).
3. The extracted pixels are used to train a 2-degree polynomial model.

Here is an example of the detected lane pixels and the fitted polynomial models. Other examples can be found in `./output_images/fitted/`.
![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

To calculate the radius of curvature, we use the equation
```python
def calc_radius_of_curvature(img, params, m_per_pixel):
    y_max = img.shape[0]
    return ((1+(2*params[0]*y_max*m_per_pixel + params[1])**2)**1.5)/np.abs(2*params[0])
```
One important thing to note here is that the `params` should be the one corresponding to the unit metres. In the code, such parameters are obtained by converting the pixel-unit-based parameters to metre-based parameters. For example,
```python
left_quad_coeff_m = np.zeros(3)
left_quad_coeff_m[0] = left_quad_coeff[0]*(1/ym_per_pixel)**2*(xm_per_pixel)
left_quad_coeff_m[1] = left_quad_coeff[1]*(xm_per_pixel/ym_per_pixel)
left_quad_coeff_m[2] = left_quad_coeff[2]*xm_per_pixel
l_roc_m = calc_radius_of_curvature(warped,left_quad_coeff_m, ym_per_pixel) 
```

Once the metre-based params are calculated, it can be simply fed into the function above. The radius of curvature will then be in metres.

For more information, see line 107 to 117 in the function `process_image()` in the Jupyter notebook `code_project.ipynb`.

To find the distance from the centre of the lane, we find the midpoint (closest to the vehicle) between the left line and right line. We then subtract this midpoint from half of the image width to find the deviation from the lane centre in metres. Negative values are taken to be deviation to the left while positive values denote deviation to the right.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 141 through 168 in my code in `code_project.ipynb` in the function `process_img()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_videos/project_video_output.mp4).
Video results for other more difficult inputs can be found in `./output_videos/`.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

While the pipeline worked fine on `project_video.mp4`, it is not working that well for more difficult videos like `challenge_video.mp4` and `harder_challenge_video.mp4`. The issues are listed here.

#### Issue 1
Owing to the use of line averages in the line detection algorithm, there is an inherent latency in the detected lines reflecting the current frame. This problem is compounded by the fact that only lines which are good are accumulated into the averages. Good in this case refers to instantaneous line estimates not deviating too siginicantly from the previous frame, and the case where the instantaneous estimates for both the right and left line being approximately parallel. The latter is checked by computing the second derivative and comparing that the values for both right and left are within a certain threshold. Only when the two aforementioned conditions are met that the estimates are added to the averages. The latency introduces problems when the lane curves really rapidly, i.e. a windy road as seen in `harder_challenge_video.mp4`.

Possible solutions to this problem are:  
* Decrease the window size for calculating the line averages. However, this can't be too small lest it introduces jitter.
* Better outlier rejection in the fitting of the polynomial. For example, a RANSAC polynomial fitting algorithm could be used.
* Use of velocity information and current steering angle (from the CAN bus) to properly predict the plausible lane markings in the next frame. The prediction can more accurately guide the search of pixels in the next frame, instead of just using plain convolution and window search of line pixels around the previously fitted lines.

#### Issue 2
In the two challenge videos, it is often times difficult to see where the line markings are visually due to sunlight glare. This poses a problem for the algorithm implemented in the project. More intelligent combination of thresholding and colour-space conversion are required to rectify this issue. One example is to quantify the contrast level of the overall frame. Depending on the contrast profile, perform a different combination of thresholding and colour-space conversion. Objects can also be segmented from the background to prevent cars and motocycles in the scene from being considered in the search for pixels.
