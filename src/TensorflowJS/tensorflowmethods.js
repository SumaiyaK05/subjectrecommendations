import * as tf from "@tensorflow/tfjs";

import { OPTIMIZERS } from "./optimisers";

export function DummyData() {
    // 14/9/2022: currently Subject data non existent. Using bespoke dummy data to get the POC model running in the react app
    // 
    const input = tf.tensor([0, 2, 4, 7, 10, 20, 50, 100], [8, 1]); //number of values,
    const label = tf.tensor([5, 9, 13, 19, 25, 45, 105, 210], [8, 1]);
    return [input, label];
  }

  export function createMLModel({
    //parameters
    units = 1,
    learningRate = 0.01,
    optimizer = "adam",
  }) {
    const selectOptimizer = (optimizer) => {
      return OPTIMIZERS[optimizer].fn(learningRate);
    };
    //create the model object
    const model = tf.sequential();
    model.add(tf.layers.dense({ units, inputShape: [1] }));
    model.compile({
        //optimiser - searches the most accurate form possible for the model
      optimizer: selectOptimizer(optimizer),
      //loss function
      loss: "meanSquaredError",
    });
    return model;
  }

//actually train the model
//params:
//model: TF Model OBject
//input: input data (array)
//label: label data (array)
//epochs: int (default 150)
export async function trainModel(model, input, label, epochs = 150) {
    await model.fit(input, label, { epochs });
  }