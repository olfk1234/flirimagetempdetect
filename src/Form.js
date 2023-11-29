import React, {useState} from "react";
import { Navbar } from "./Navbar";
import axios from "axios";
export function Form(){

        const [imageFile, setImageFile] = useState(null);
        const [minTemperature, setMinTemperature] = useState(null);
        const [maxTemperature, setMaxTemperature] = useState(null);
        const [temperatureUnit, setTemperatureUnit] = useState(null);
        const [temperatureUnitOutput, setTemperatureUnitOutput] = useState(null);
        const [image, setImage] = useState(null);
        const [disable, setDisable] = useState(null);

      
        const handleImageChange = (event) => {
          const file = event.target.files[0];
          setImageFile(file);
        };
      
        const handleMinTemperatureChange = (event) => {
          setMinTemperature(event.target.value);
        };
      
        const handleMaxTemperatureChange = (event) => {
          setMaxTemperature(event.target.value);
        };
      
        const handleTemperatureUnitChange = (event) => {
          setTemperatureUnit(event.target.value);
        };

        const handleTemperatureUnitChangeOutput = (event) => {
          setTemperatureUnitOutput(event.target.value);
        };
      
        const handleSubmit = async (event) => {
          event.preventDefault();
          const formData = new FormData();
          formData.append('imageFile', imageFile);
          formData.append('minTemperature', minTemperature);
          formData.append('maxTemperature', maxTemperature);
          formData.append('temperatureUnit', temperatureUnit);
          formData.append('temperatureUnitOutput', temperatureUnitOutput);

          setDisable(true)
        return axios.post('http://localhost:5000/upload', formData, {
            headers: {
            'Content-Type': 'multipart/form-data'
            }
        }).then(() => {

          axios.get('http://localhost:5000/');
          setImage(true);
  
        });
        
        };
  
    return(
      <div>
        <Navbar/>
        <div style = {{textAlign: "center"}}>
        <h1>FLIR Thermal Image Detector</h1>
        <p>Put an image from a FLIR device and get the minimum and maximum values from the FLIR Tools app</p>
        </div>
        <form onSubmit= {(e) => {handleSubmit(e)}} style = {{textAlign: "center"}}>
        <label for="imageFile" style = {{paddingRight: "20px"}}>Select an image file:</label>
        <br/>
        <input onChange= {(e) => {handleImageChange(e)}} disabled = {disable} style= {{marginLeft: "5rem"}} type="file" id="imageFile" name="imageFile" accept="image/jpeg, image/jpg, image/png, image/gif" required = {true}/>
        <br/>
        <br/>
        <label>Minimum Temperature: </label>
        <input onChange= {(e) => {handleMinTemperatureChange(e)}} disabled = {disable} type= "number" style= {{marginLeft: "4.5px", width: "60px"}} step = "0.01" equired = {true}/>
        <br/>
        <label>Maximum Temperature: </label>
        <input onChange= {(e) => {handleMaxTemperatureChange(e)}} disabled = {disable} type= "number" style= {{marginLeft: "4.5px", width: "60px"}} step = "0.01" required = {true}/>
        <br/>
        <br/>
        <label>Units of Input Minimum and Maximum Temperature</label>
        <br/>
        <input onChange= {(e) => {handleTemperatureUnitChange(e)}} disabled = {disable} type="radio" id="option1" name="myGroup" value="Celsius"/>
        <label for="option1">Celsius</label>
        <br/>
        <input onChange= {(e) => {handleTemperatureUnitChange(e)}} disabled = {disable} type="radio" id="option2" name="myGroup" value="Fahrenheit" style = {{marginLeft: "29px"}}/>
        <label for="option2">Fahrenheit</label>
        <br/>
        <br/>
        <label>Desired Units of Output Temperature</label>
        <br/>
        <input onChange= {(e) => {handleTemperatureUnitChangeOutput(e)}} disabled = {disable} type="radio" id="option1o" name="myGroup1" value="Celsius"/>
        <label for="option1o">Celsius</label>
        <br/>
        <input onChange= {(e) => {handleTemperatureUnitChangeOutput(e)}} disabled = {disable} type="radio" id="option2o" name="myGroup1" value="Fahrenheit" style = {{marginLeft: "29px"}}/>
        <label for="option2o">Fahrenheit</label>
        <br/>
        <br/>
        <button type="submit" disabled = {imageFile == null || minTemperature == null || maxTemperature == null || temperatureUnit == null || temperatureUnitOutput == null}>Submit</button>
        <button onClick={()=>{window.location.reload()}}>Reset</button>
        <p hidden={!image}>Results:</p>
        {image && <img src="http://localhost:5000/static/img.jpg" alt="ML Output" />}
        </form>
        </div>
    )

}

