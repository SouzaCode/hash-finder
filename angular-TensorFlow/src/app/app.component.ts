import { Component } from '@angular/core';
import { setTheme } from 'ngx-bootstrap/utils';
import sha256 from 'crypto-js/sha256';
import * as tf from '@tensorflow/tfjs';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css'],
})
export class AppComponent {
  title = 'angular-TensorFlow';
  inputData: Number;
  linearModel: tf.Sequential;
  prediction: any;
  hashdata: any;
  hashPredict: any;
  constructor() {
    setTheme('bs3');
  }

  ngOnInit() {
    this.trainNewModel();
  }
  async trainNewModel() {
    console.log('Começou a treinar');

    this.linearModel = tf.sequential();
    this.linearModel.add(tf.layers.dense({ units: 1, inputShape: [1] }));
    //this.linearModel.add(tf.layers.dense({ units: 1, inputShape: [1] }));
    //this.linearModel.add(tf.layers.dense({ units: 1 }));
    this.linearModel.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });
    let trainData = [];
    let trainOutput = [];
    for (let i = 0.0; i < 10; i++) {
      let hash = sha256(i);
      let decimal = parseInt(hash, 16) / 10 ** 70;
      //console.log(decimal);

      trainOutput.push(parseFloat(i.toString()));
      trainData.push(hash);
    }
    console.log(trainData);
    console.log(trainOutput);

    const xs = tf.tensor1d(trainData);
    const ys = tf.tensor1d(trainOutput);
    await this.linearModel.fit(xs, ys);

    console.log('treinado!');
  }

  linearPrediction(val: number, hash) {
    console.log(val);
    console.log(val.toString(16));

    const output = this.linearModel.predict(tf.tensor2d([hash], [1, 1])) as any;

    this.hashPredict = Number(Array.from(output.dataSync())[0]).toString(16);
    /*
    this.hashPredict = this.prediction.toString().split('.')[0];

    console.log(this.hashPredict);
    for (let i = 0; i < 70; i++) {
      this.hashPredict = this.hashPredict + '0';
    }
    this.hashPredict = Number(this.hashPredict).toString(16);
    console.log(Number(this.hashPredict).toString(16));
    */
  }
  handleChange(data: number): void {
    this.inputData = data;
    this.hashdata = sha256(data);
    let decimal = parseInt(this.hashdata, 16);
    console.log(decimal);
    console.log(parseInt(decimal.toString(16), 16));

    this.linearPrediction(decimal, this.hashdata);
  }
}
