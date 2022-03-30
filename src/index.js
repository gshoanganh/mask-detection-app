import React from "react";
import ReactDOM from "react-dom";
import * as tf from '@tensorflow/tfjs';
import { loadGraphModel } from '@tensorflow/tfjs-converter';
import "./styles.css";
tf.setBackend('webgl');

const threshold = 0.85;

async function load_model() {
  // It's possible to load the model locally or from a repo
  // You can choose whatever IP and PORT you want in the "http://127.0.0.1:8080/model.json" just set it before in your https server
  const model = await loadGraphModel("http://127.0.0.1:8080/model.json");  
  return model;
}

let classesDir = [{ 'name': 'MaskWhite', 'id': 1, 'color': 'red' },
{ 'name': 'MaskBlue', 'id': 2, 'color': 'blue' }, 
{ 'name': 'NoMask', 'id': 2, 'color': 'black' }]

class App extends React.Component {
  videoRef = React.createRef();
  canvasRef = React.createRef();


  componentDidMount() {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      const webCamPromise = navigator.mediaDevices
        .getUserMedia({
          audio: false,
          video: {
            facingMode: "user"
          }
        })
        .then(stream => {
          window.stream = stream;
          this.videoRef.current.srcObject = stream;
          return new Promise((resolve, reject) => {
            this.videoRef.current.onloadedmetadata = () => {
              resolve();
            };
          });
        });

      const modelPromise = load_model();

      Promise.all([modelPromise, webCamPromise])
        .then(values => {
          this.detectFrame(this.videoRef.current, values[0]);
        })
        .catch(error => {
          console.error(error);
        });
    }
  }

  detectFrame = (video, model) => {
    tf.engine().startScope();
    model.executeAsync(this.process_input(video)).then(predictions => {
      this.renderPredictions(predictions, video);
      requestAnimationFrame(() => {
        this.detectFrame(video, model);
      });
      tf.engine().endScope();
    });
  };

  process_input(video_frame) {
    const tfimg = tf.browser.fromPixels(video_frame).toInt();
    const expandedimg = tfimg.transpose([0, 1, 2]).expandDims();
    return expandedimg;
  };

  buildDetectedObjects(scores, threshold, boxes, classes, classesDir) {
    //console.log('detection: ', scores, threshold)
    const detectionObjects = []
    var video_frame = document.getElementById('frame');

    scores[0].forEach((score, i) => {

      if (score > threshold) {

        //console.log('detection: ----------------', score, i, classesDir[classes[i] - 1].name, score.toFixed(4))
        let [y, x, h, w] = boxes[0][i]
        //console.log('boxes y, x, h, w: ', boxes[0][i])

        const bbox = [];

        let x1 = x * video_frame.offsetWidth;
        let y1 = y * video_frame.offsetHeight;
        let x2 = w * video_frame.offsetHeight;
        let y2 = h * video_frame.offsetWidth;

        bbox[0] = x1;
        bbox[1] = y1;
        bbox[2] = x2 - x1;
        bbox[3] = y2 - y1;

        detectionObjects.push({
          class: classes[i],
          label: classesDir[classes[i] - 1].name,
          color: classesDir[classes[i] - 1].color,
          score: score.toFixed(4),
          bbox: bbox
        })
      }
    })
    return detectionObjects
  }

  renderPredictions = predictions => {
    const ctx = this.canvasRef.current.getContext("2d");
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    // Font options.
    const font = "16px sans-serif";
    ctx.font = font;
    ctx.textBaseline = "top";

    //Getting predictions
    // const boxes = predictions[4].arraySync();
    // const scores = predictions[5].arraySync();
    // const classes = predictions[6].dataSync();
    const boxes = predictions[7].arraySync();
    const scores = predictions[1].arraySync();
    const classes = predictions[5].dataSync();

    console.log('- predictions: 0', predictions[0].arraySync())
    //console.log(' predictions: 7', predictions[7].arraySync())
    //console.log(' predictions: 4', predictions[4].arraySync())

    const detections = this.buildDetectedObjects(scores, threshold,
      boxes, classes, classesDir);

    detections.forEach(item => {
      const [x, y, w, h] = item.bbox
      console.log('item: ', item.bbox)

      // Draw the bounding box.
      ctx.strokeStyle = item.color //"#00FFFF";
      ctx.lineWidth = 4;
      ctx.strokeRect(x, y, w, h);

      // Draw the label background.
      ctx.fillStyle = "#00FFFF";
      const textWidth = ctx.measureText(item["label"] + " " + (100 * item["score"]).toFixed(2) + "%").width;
      const textHeight = parseInt(font, 10); // base 10
      ctx.fillRect(x, y, textWidth + 4, textHeight + 4);

    });

    detections.forEach(item => {
      const x = item['bbox'][0];
      const y = item['bbox'][1];

      // Draw the text last to ensure it's on top.
      ctx.fillStyle = "#000000";
      ctx.fillText(item["label"] + " " + (100 * item["score"]).toFixed(2) + "%", x, y);
    });
  };

  render() {
    return (
      <div>
        <h1>Real-Time Object Detection: Mask</h1>
        <h3>MobileNet V2</h3>
        <video
          style={{ height: '600px', width: "500px" }}
          className="size"
          autoPlay
          playsInline
          muted
          ref={this.videoRef}
          width="600"
          height="500"
          id="frame"
        />
        <canvas
          className="size"
          ref={this.canvasRef}
          width="600"
          height="500"
        />
      </div>
    );
  }
}

const rootElement = document.getElementById("root");
ReactDOM.render(<App />, rootElement);
