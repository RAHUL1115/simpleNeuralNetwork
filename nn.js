class NeuralNetwork{
  constructor(inputNodes,hiddenNodes,outputNodes,learningRate){
    if (inputNodes instanceof NeuralNetwork) {
      let preNuralNetwork = inputNodes;
      this.inputNodes = preNuralNetwork.inputNodes;
      this.hiddenNodes = [...preNuralNetwork.hiddenNodes];
      this.outputNodes = preNuralNetwork.outputNodes;
      this.learningRate = preNuralNetwork.learningRate || 0.1;
      this.wHH = preNuralNetwork.wHH.map(hiddenWeights=>{
        return hiddenWeights.clone();
      });
      this.wHO = preNuralNetwork.wHO.clone();
    } else if (inputNodes instanceof Object) {
      let preJsonData = inputNodes;
      this.inputNodes = preJsonData.inputNodes;
      this.hiddenNodes = [...preJsonData.hiddenNodes];
      this.outputNodes = preJsonData.outputNodes;
      this.learningRate = preJsonData.learningRate || 0.1;
      this.wHH = preJsonData.wHH.map(hiddenWeights => {
        return nj.array(hiddenWeights);
      });
      this.wHO = nj.array(preJsonData.wHO);
    } else{
      this.inputNodes = inputNodes;
      this.hiddenNodes = hiddenNodes;
      this.outputNodes = outputNodes;
      this.learningRate = learningRate || 0.1;

      // ! calculate all hidden weight
      this.wHH = [];
      for (let i = 0; i < this.hiddenNodes.length; i++) {
        // check if multilayer or not
        if (i == 0) {
          this.wHH.push(nj.random([this.inputNodes, this.hiddenNodes[0]]).subtract(0.5))
        } else {
          this.wHH.push(nj.random([this.hiddenNodes[i - 1], this.hiddenNodes[i]]).subtract(0.5))
        }
      }

      // ! calculate hidden to output
      this.wHO = nj.random([this.hiddenNodes[this.hiddenNodes.length - 1], this.outputNodes]).subtract(0.5)
    }
  }
  
  // ! sigmoid activation function
  activationFunction(x){
    return nj.sigmoid(x);
  }

  // ! sigmoid deactivation function
  deactivationFunction(x){
    return x.multiply(x.subtract(1.0)).multiply(-1);
  }

  train(inputList,targetList){
    let inputs = nj.array(inputList);
    let targets = nj.array(targetList);
    inputs = (inputs.ndim <= 1) ? nj.array([inputList]) : inputs;
    targets = (targets.ndim <= 1) ? nj.array([targetList]) : targets;

    // * feed forward
    // ? calculate all hidden inputs and outputs
    let hiddenInputs = [];
    let hiddenOutputs = [];
    for (let i = 0; i < this.wHH.length; i++) {
      if (i == 0) {
        hiddenInputs.push(nj.dot(inputs, this.wHH[i]))
        hiddenOutputs.push(this.activationFunction(hiddenInputs[hiddenInputs.length - 1]))
      } else {
        hiddenInputs.push(nj.dot(hiddenOutputs[i - 1], this.wHH[i]))
        hiddenOutputs.push(this.activationFunction(hiddenInputs[hiddenInputs.length - 1]))
      }
    }
    
    // ? calculate all hidden to outputs
    let finalInputs = nj.dot(hiddenOutputs[hiddenOutputs.length - 1], this.wHO);
    let finalOutputs = this.activationFunction(finalInputs);
    
    // * error calculation
    // ? error to output
    let outputError = targets.subtract(finalOutputs)

    // ? error of hidden
    let hiddenError = []
    for (let i = 0; i < this.wHH.length; i++) {
      if (i == 0) {
        hiddenError.push(nj.dot(outputError, this.wHO.T))
      } else {
        hiddenError.push(nj.dot(hiddenError[hiddenError.length - i], this.wHH[this.wHH.length - i].T))
      }
    }
    hiddenError = hiddenError.reverse()

   
    // * error reduction
    // ? hidden error reduction
    for (let i = 0; i < this.wHH.length; i++) {
      if (i == 0) {
        this.wHH[i] = this.wHH[i].add(nj.dot(inputs.T, hiddenError[i].multiply(this.deactivationFunction(hiddenOutputs[i]).multiply(this.learningRate))));
      } else {
        this.wHH[i] = this.wHH[i].add(nj.dot(hiddenOutputs[i - 1].T, hiddenError[i].multiply(this.deactivationFunction(hiddenOutputs[i]).multiply(this.learningRate))));
      }
    }

    // ? output error reduction
    this.wHO = this.wHO.add(nj.dot(hiddenOutputs[hiddenOutputs.length - 1].T, outputError.multiply(this.deactivationFunction(finalOutputs).multiply(this.learningRate))));
  }

  query(inputList){
    let inputs = nj.array(inputList);
    inputs = (inputs.ndim <= 1) ? nj.array([inputList]) : inputs;

    let hiddenInputs = [];
    let hiddenOutputs = [];
    for (let i = 0; i < this.wHH.length; i++) {
      if (i == 0) {
        hiddenInputs.push(nj.dot(inputs, this.wHH[i]))
        hiddenOutputs.push(this.activationFunction(hiddenInputs[hiddenInputs.length - 1]))
      } else {
        hiddenInputs.push(nj.dot(hiddenOutputs[i - 1], this.wHH[i]))
        hiddenOutputs.push(this.activationFunction(hiddenInputs[hiddenInputs.length - 1]))
      }
    }

    // ? calculate all hidden to outputs
    let finalInputs = nj.dot(hiddenOutputs[hiddenOutputs.length - 1], this.wHO);
    let finalOutputs = this.activationFunction(finalInputs);

    return finalOutputs;
  }

  serialize() {
    let json = JSON.parse(JSON.stringify(Object.assign({}, this)));
    json.wHH = this.wHH.map(hiddenWeights => {
      return hiddenWeights.clone().tolist();
    });
    json.wHO = this.wHO.clone().tolist();
    return json;
  }

  // ? adding code for nuroevolution
  copy(){
    return new NeuralNetwork(this)
  }

  mutate(rate){
    this.wHH.forEach(hidenWeights => {
      for (let i = 0; i < hidenWeights.shape[0]; i++) {
        for (let j = 0; j < hidenWeights.shape[1]; j++) {
          if(Math.random() < rate){
            // hidenWeights.set(i, j, (Math.random() * 2 - 1));
            hidenWeights.set(i, j, (hidenWeights.get(i,j) + randomGaussian(0,0.1)));
          }
        }
      }
    });
    for (let i = 0; i < this.wHO.shape[0]; i++) {
      for (let j = 0; j < this.wHO.shape[1]; j++) {
        if (Math.random() < rate) {
          // this.wHO.set(i, j, (Math.random() * 2 - 1));
          this.wHO.set(i, j, (this.wHO.get(i, j) + randomGaussian(0, 0.1)));
        }
      }
    }
  }
}

// let a;
// a = new NuralNetwork(2, [4], 1, 0.1)
// for (let index = 0; index < 10000; index++) {
//   a.train([
//     [1, 0],
//     [0, 1],
//     [1, 1],
//     [0, 0]
//   ], [
//     [1],
//     [1],
//     [0],
//     [0]
//   ])
// }
// a.query([1, 0]);