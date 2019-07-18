# BackgroundSubtraction
Stauffer Grimson Background Subtraction

This is the submission for the Stauffer Grimson Background Subtraction, submitted by 
Shreyansh Dixit; and
Manish Minocha
towards fulfillment of the ELL784 course at IIT Delhi (Introduction to Machine Learning)

APPROACH:
* Programming Language: We chose Python as the programming language of choice because of the user support and functionality provided by the libraries available. We did expect python to be slightly slower than C, and hence chose some algorithmic optimzations highlighted below. Plus, since python was a new language for both of us, hence we saw this as an opportunity to learn something more.
* Greyscale Images and Videos: Just to save the computing power required, we have performed the modelling on greyscale frames of the video. Hence the output is a greyscale video as well. On hindsight, we could have got a color RGB video output using greyscale computation, since we assume the R,G,B values to be similarly correlated with greyscale values.
* Approximations on K Mean clustering: Since K means is used for initialiation only, we took some liberties in reducing computational requirements. For example, since the library functions of K means clustering (from scikit / sklearn python libraries) did not have inbuilt functions of returning standard deviation, we approximated the SD to be the same across all clusters. This could be the reason why the initial frames of our model seem to be out-of-sync with the expected output.
The rest of the code is a verbatim implementation of the Stauffer Grimson Background subtraction model.

DESCRIPTION OF THE CODE:
The main subroutines of the code are described below
* extractFrames(): get the frames one-by-one, convert to greyscale, and then invoke the subroutine to process the frames. We used openCV to split the frames
* intializeModel(): K Means clustering of the first frame. We used sklearn python library for K Means.
* processFrame(): perform the Gaussian Multi-mixture model for each frame, and try to fit the pixels as either background or foreground
* compileVideo(): Convert the background and foreground frames into the respective videos.

OBSERVATIONS AND CONCLUSIONS:
* Changing value of K (number of gaussians): We tried different values of K (3 to 5). Changing value of K did not seem to effect the output much probably our method of grescaling used.
* Changing value of alpha (learning constant): We tried changing value of alpha from low (0.001) to higher (0.03). Lower value of alpha was not conducive and resulted in a very noisy foreground.
* Time take to process: We tried multiple workstation configurations to run the program. The time taken is summarised below: 
	On Google Cloud ( micro - 1 vcpu / 0.6 GB vRAM): Time taken = 215 minutes (approx 3.5 hours)
	On pythonanywhere.com : Could not execute with the computing power purchased i.e. the code is too intensive to run economically on pythonanywhere.com
	On local laptop ( intel i5, 4 CPU, 4 GB vRAM): Time taken: Approx 75 minutes ( 1 hour 15 min)

FUTURE THOUGHTS:
Quite evidently, the video output we generated is not as smooth or clear as the other output we have seen. We tried a few combinations of parameters (which time permitted). However, what is more intriguing is whether we can build a model to determine the quality of the output generated. 
* What we would we define as quality of background video
Of course we would need to define what we mean by quality, but in more intuitive terms background video quality would be to see how soon we are able to get frames whose pixels lie within a threshold SD of 1 gaussian (or 2 gaussians max).
* What we would define as quality of foreground video
That is a little more complex. No intuitive definition of  quality of foreground video. Even though we can use consistency of ratio of BG pixels (static) to foreground pixels (changing) to determine foreground video quality, but this definition will fail in case an object is moving out of the video (e.g. a car in this case)
