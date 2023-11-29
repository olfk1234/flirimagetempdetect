import React from "react";
import { Navbar } from "./Navbar";
export function HowToUse(){
return(
    <div style = {{textAlign: "center"}}>
        <Navbar/>
        <div style={{marginLeft : "15rem", marginRight: "15rem"}}>
        <h1>How to Use</h1>
        <p>Instructions for FLIR ONE but valid for any FLIR device able to get a minimum temperature, a maximum temperature, and a thermal image </p>
        <ol style = {{textAlign: "left"}}>
            <li>Take thermal image with a FLIR device</li>
            <li>Upload image to FLIR tools</li>
            <li>Enter the editing tools by tapping on the image and then tapping on the pencil in the top right corner</li>
            <li>Get the maximum and minimum temperature of the image by looking at the scale on the right side</li>
            <li>Set the MSX slider at the bottom to 0 to convert the image to a pure thermal image</li>
        </ol>
        </div>
    </div>
)
}