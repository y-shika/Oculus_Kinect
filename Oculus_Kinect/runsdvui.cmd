cd /d "C:\Users\Yusuke Shikanai\source\repos\Oculus_Kinect\Oculus_Kinect" &msbuild "Oculus_Kinect.vcxproj" /t:sdvViewer /p:configuration="Release" /p:platform=x64
exit %errorlevel% 