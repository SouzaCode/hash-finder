import { Component } from '@angular/core';
import { setTheme } from 'ngx-bootstrap/utils';

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
    this.linearModel.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });
    let trainData = [3.2, 4.4, 5.5, 6.6, 7.7];
    let trainOutput = [1.6, 2.7, 2.9, 3.19, 1.684];

    const xs = tf.tensor1d(trainData);
    const ys = tf.tensor1d(trainOutput);
    await this.linearModel.fit(xs, ys);

    console.log('treinado!');
  }

  linearPrediction(val: number) {
    const output = this.linearModel.predict(
      tf.tensor2d([Number(val)], [1, 1])
    ) as any;

    this.prediction = Array.from(output.dataSync())[0];
  }
  handleChange(data: number): void {
    this.inputData = data;
    this.linearPrediction(data);
  }
}
