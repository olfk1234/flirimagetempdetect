import React from "react";
import { Navbar } from "./Navbar";
export function Explanation(){
return(
    <div style = {{textAlign: "center"}}>
        <Navbar/>
        <div style={{marginLeft : "15rem", marginRight: "15rem"}}>
        <h1>Explanation of Algorithms and Processes</h1>
        <h2>Temperature Detection</h2>
        <p>When the maximum and minimum temperatures are given, linear interpolation can be used to determine the temperature of a specific pixel in a thermal image based on the intensity value. The initial thermal image is converted to a normalized grayscale image array to determine the approximate intensity values. The user provides the minimum and maximum temperature in the image using the FLIR Tools App. From this, the highest intensity can be assigned to the highest temperature and the lowest intensity can be assigned to the lowest temperature. The rest of the temperatures in the image can be determined based on the approximate intensity values.</p>
        <br/>
        <h2>Object Detection</h2>
        <p>The matrix for object detection tells the temperature detection algorithm where to focus its processes. <b>Yolov8</b>  was trained on a Kaggle dataset of over 1000 thermal images of faces with different colornaps and allows for focused human temperature detection.</p>
        </div>
    </div>
)
}