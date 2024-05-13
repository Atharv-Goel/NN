/*
 * Creator: Atharv Goel
 * Started: January 5, 2024
 *
 * Description:
 * This module aims to act as an N-Layer Neural Network with backpropagation.
 * It essentially works by repeatedly feeding forward the current weights through
 * the network to calculate a single predicted outcome. Using gradient descent, it
 * takes the predicted outcome and the expected outcome to modify the weights to
 * attempt and improve the predicted outcome in future runs of the network. The
 * backpropagation is used to speed up the training of the neural network.
 */

/*
 * TABLE OF CONTENTS
 *
 * sigmoid
 * sigmoidDer
 * tanhDer
 * f
 * fDer
 * random
 * assignCase
 * printCases
 * assign
 * setConfig
 * echoConfig
 * allocate
 * loadWeights
 * saveWeights
 * loadCases
 * populate
 * feedForwardRun
 * feedForwardTrain
 * gradientDescent
 * run
 * train
 * main
 */

#include <iostream>
#include <random>
#include <string>
#include <fstream>
#include <chrono>
#include <cmath>

int numLayers; // Number of layers (including the output layer)
int* nodes;    // Number of nodes in each layer
/*
 * NOTE: nodes[0] will be used to refer to the number of nodes in the input layer
 * (since 0 points to the first layer) and nodes[numLayers - 1] will be used to
 * refer to the number of nodes in the output layer (since numLayers - 1 is the
 * last layer)
 */

int numCases; // The number of training cases
int** cases;  // The training cases themselves
bool print; // Whether to print the cases or not

double low;  // Lower random bound for weights
double high; // Upper random bound for weights

double (*threshold) (double); //Pointer to the function that will be used as the threshold function
double (*thresholdDer) (double); //Pointer to the function that will be used as the threshold function derivative

double lambda;  // The Learning factor
int iterations; // Max number of iterations (terminating condition)
double error;   // Max error (terminating condition)
int keepAlive;  // Number of iterations between total error message printing out

bool mode = false; // Training (false) vs Running (true) Mode

double** a; // The nodes (including the output layer)
double* T;  // Expected outputs

double*** w;    // The weights
double** theta; // Layers of dot products

double** psi; // Stores node-specific doubles required in the calculations

std::string loadLocation; // The file in which to load the weights from
std::string saveLocation; // The file in which to save the weights to
std::string casesLocation; // The file in which the cases are stored

/*
 * Returns the same double that was passed in. This function is simple enough that
 * there are not any special cases that need to be taken into consideration.
 */
double identity(double x)
{
   return x;
}

/*
 * Returns the derivative of the double (which is just 1). This function is simple enough that
 * there are not any special cases that need to be taken into consideration.
 */
double identityDer(double x)
{
   return 1.0;
}


/*
 * Returns the sigmoid of a double. This function is simple enough that
 * there are not any special cases that need to be taken into consideration.
 */
double sigmoid(double x)
{
   return 1.0 / (1.0 + exp(-x));
}

/*
 * Returns the derivative of sigmoid of a double. This function is simple enough that
 * there are not any special cases that need to be taken into consideration.
 */
double sigmoidDer(double x)
{
   double temp = sigmoid(x);
   return temp * (1.0 - temp);
}

/*
 * Returns the derivative of tanh for a double. The math library has a built-in
 * tanh function that can be directly used.
 */
double tanhDer(double x)
{
   double temp = std::tanh(x);
   return 1 - temp * temp;
}

/*
 * This function is the threshold function used in the neural network.
 * It uses the function specified in the config file.
 */
double f(double x)
{
   return threshold(x);
}

/*
 * This function is the derivative of the threshold function.
 * It uses the function specified in the config file.
 */
double fDer(double x)
{
   return thresholdDer(x);
}

/*
 * This is a random double generator used mainly for the weights initialization
 * at the beginning of the training process. It simply returns a double between
 * the given lower and upper bounds. The random device and generator are declared
 * outside the function itself, since there is no point in recreating the same
 * thing for each new random double required.
 */
std::random_device rd;
std::mt19937 gen(rd());
double random(double lower, double upper)
{
   std::uniform_real_distribution<double> dis(lower, upper);
   return dis(gen);
}

/*
 * This function just assigns the input node vales and the expected value based
 * on the case number provided to it.
 */
void assignCase(int caseNumber)
{
   for (int m = 0; m < nodes[0]; m++)  //Assign cases to the first layer (the input nodes)
      a[0][m] = (double) cases[caseNumber][m];

   for (int i = 0; i < nodes[numLayers - 1]; i++) // Assign cases to the expected outputs
      T[i] = (double) cases[caseNumber][nodes[0] + i];

   return;
}

/*
 * This method prints out the test cases used for training and running.
 */
void printCases()
{
   if (print) // Only print the cases if specified in the config file
   {
      std::cout << "Truth Table:" << std::endl << std::endl;
      for (int c = 0; c < numCases; c++)
      {
         std::cout << "Inputs: ";
         for (int m = 0; m < nodes[0]; m++)
            std::cout << cases[c][m] << ' ';

         std::cout << "| Outputs: ";
         for (int i = 0; i < nodes[numLayers - 1]; i++)
            std::cout << cases[c][nodes[0] + i] << ' ';

         std::cout << std::endl;
      }
      std::cout << std::endl << "-----------------------------------" << std::endl << std::endl;
   } // if (print)

   return;
}

/*
 * This method is a helper method for setConfig. It reads through
 * directive lines, using a certain identifier to set specific config
 * variables to the correct value in the file.
 */
void assign(std::string& directive, std::ifstream& file)
{
   /*
    * Usually this would be a case construct, but c++ does not support
    * strings inside of case statements.
    */

   if (directive == "LAYERS")
   {
      file >> numLayers;
      nodes = new int[numLayers];
   }

   else if (directive == "SIZE")
   {
      for (int alpha = 0; alpha < numLayers; alpha++)
         file >> nodes[alpha];
   }

   else if (directive == "CASES")
      file >> numCases;

   else if (directive == "CASELOC")
      file >> casesLocation;

   else if (directive == "PRINT")
      file >> print;

   else if (directive == "LOWER")
      file >> low;

   else if (directive == "UPPER")
      file >> high;

   else if (directive == "RATE")
      file >> lambda;

   else if (directive == "ITERS")
      file >> iterations;

   else if (directive == "ERROR")
      file >> error;

   else if (directive == "STATUS")
      file >> keepAlive;

   else if (directive == "MODE")
      file >> mode;

   else if (directive == "LOAD")
      file >> loadLocation;

   else if (directive == "SAVE")
      file >> saveLocation;

   else if (directive == "THRESHOLD")
   {
      int temp;
      file >> temp;
      switch (temp)
      {
         case 0:
            threshold = sigmoid;
            thresholdDer = sigmoidDer;
            break;
         case 1:
            threshold = std::tanh;
            thresholdDer = tanhDer;
            break;
         case 2:
            threshold = identity;
            thresholdDer = identityDer;
            break;
         default:
            std::cout << "Invalid threshold choice, using sigmoid" << std::endl;
            threshold = sigmoid;
            thresholdDer = sigmoidDer;
            break;
      } // switch (temp)

   }

   else
      std::cout << "Unknown directive: " << directive << std::endl;

   return;
} //void assign(std::string& directive, std::ifstream& file)

/*
 * This method sets all the configuration parameters that are necessary to train
 * or run the network. It pulls its info from a file passed into this function.
 */
void setConfig(const std::string &configFile)
{
   std::ifstream config(configFile); // Load the config file
   std::string directive;

   while (config.is_open() && config) // Make sure the file is open and that the end of the file has not been reached
   {
      if (config.get() == '#') // Directive lines begin with a #
      {
         config >> directive;
         assign(directive, config);
         std::getline(config, directive);
      } // if (config.get() == '#')

   } // while (config.is_open() && config)
   config.close();

   return;
} // setConfig(std::string configFile)

/*
 * Prints out the config variables for the user to see.
 */
void echoConfig()
{
   std::cout << "This is an N-Layers Neural Network" << std::endl;
   std::cout << "The size of each layer is: ";
   for (int alpha = 0; alpha < numLayers; alpha++)
      std::cout << nodes[alpha] << " ";
   std::cout << std::endl << std::endl;

   if (loadLocation.empty())
      std::cout << "Weights initialized randomly between " << low << " and " << high << std::endl;
   else
      std::cout << "Weights being initialized from " << loadLocation << std::endl;
   std::cout << std::endl;

   if (mode == 0)
   {
      std::cout << "Training Parameters:" << std::endl;
      std::cout << "Maximum iterations: " << iterations << std::endl;
      std::cout << "Minimum error threshold: " << error << std::endl;
      std::cout << "Learning rate: " << lambda << std::endl;
      std::cout << std::endl;
   }
   std::cout << "-----------------------------------" << std::endl << std::endl;

   return;
} // void echoConfig()

/*
 * This method allocates memory for all the arrays depending on the config
 * params defined above.
 */
void allocate()
{
   a = new double*[numLayers];
   for (int layer = 0; layer < numLayers; layer++)
      a[layer] = new double[nodes[layer]];

   T = new double[nodes[numLayers - 1]];

   cases = new int*[numCases];
   for (int c = 0; c < numCases; c++)
      cases[c] = new int[nodes[0] + nodes[numLayers - 1]];

   w = new double**[numLayers - 1];
   for (int alpha = 0; alpha < numLayers - 1; alpha++)
   {
      w[alpha] = new double*[nodes[alpha]];

      for (int beta = 0; beta < nodes[alpha]; beta++)
         w[alpha][beta] = new double[nodes[alpha + 1]];
   }

   if (!mode) // No point in allocating space for arrays if they are not going to be used
   {
      theta = new double*[numLayers - 1];
      for (int alpha = 0; alpha < numLayers - 1; alpha++)
         theta[alpha] = new double[nodes[alpha + 1]];

      psi = new double*[numLayers - 1]; // Not required for the first layer
      for (int alpha = 0; alpha < numLayers - 1; alpha++)
         psi[alpha] = new double[nodes[alpha + 1]];
   } // if (!mode)

   return;
} // void allocate()

/*
 * This is a helper method for loading weights from a file, assuming that
 * a file was provided in the config file.
 */
void loadWeights()
{
   std::fstream file(loadLocation);
   for (int alpha = 0; alpha < numLayers - 1; alpha++) // All layers except the output one
   {
      for (int beta = 0; beta < nodes[alpha]; beta++)
      {
         for (int gamma = 0; gamma < nodes[alpha + 1]; gamma++) // Weights between current layer and the next
         {
            file >> w[alpha][beta][gamma];
         }
      }
   }

   return;
} // loadWeights()

/*
 * This is a helper method for saving weights to a file, assuming that
 * a file was provided in the config file.
 */
void saveWeights()
{
   std::ofstream file(saveLocation, std::ofstream::trunc);

   for (int alpha = 0; alpha < numLayers - 1; alpha++)  // All layers except the output one
   {
      for (int beta = 0; beta < nodes[alpha]; beta++)
      {
         for (int gamma = 0; gamma < nodes[alpha + 1]; gamma++) // Weights between current layer and the next
            file << std::fixed << std::setprecision(17) << w[alpha][beta][gamma] << ' ';
         file << '\n';
      }
   } // for (int alpha = 0; alpha < numLayers - 1; alpha++)

   file.close();

   return;
} // loadWeights()

/*
 * This is a helper method for loading cases from a file, assuming that
 * a file was provided in the config file.
 */
void loadCases()
{
   std::ifstream caseFile(casesLocation);

   for (int c = 0; c < numCases && caseFile.is_open(); c++)
   {
      std::string inputFileName;// The binary file with the input weights
      caseFile >> inputFileName;
      std::ifstream inputFile(inputFileName);

      unsigned char temp; // Temporary variable to make sure only one byte is read at a time
      for (int m = 0; m < nodes[0]; m++)
      {
         inputFile >> temp;
         cases[c][m] = (int) temp;
      }
      inputFile.close();

      for (int i = nodes[0]; i < nodes[0] + nodes[numLayers - 1]; i++) // The expected outputs
      {
         caseFile >> cases[c][i];
      }

   } // for (int c = 0; c < numCases && caseFile.is_open(); c++)

   caseFile.close();

   return;
} //loadCases()

/*
 * Populates the weight and cases arrays. For weights, it uses the weight file if specified,
 * otherwise it randomly generates them.
 */
void populate()
{
   if (loadLocation.empty()) // If a load location has not been specified, use random weights
   {
      for (int alpha = 0; alpha < numLayers - 1; alpha++) // All layers except the output one
      {
         for (int beta = 0; beta < nodes[alpha]; beta++)
         {
            for (int gamma = 0; gamma < nodes[alpha + 1]; gamma++) // Weights between current layer and the next
               w[alpha][beta][gamma] = random(low, high);
         }
      }
   } // if (loadLocation.empty())

   else // If a load location has been specified, use that instead
      loadWeights();

   loadCases(); //Load the cases from the file that should have been specified in the config file

   return;
} // void initWeights()

/*
 * This function takes the current weights, and feeds them forward
 * through the network to transform the inputs into the predicted
 * outputs. It uses a threshold function to minimize the effect
 * that nodes with a negative value have. This method is not used
 * for training, so it does not store any unnecessary values.
 */
void feedForwardRun()
{
   double thetaRun; // Temporary value used for accumulating in the computations

   for (int alpha = 0; alpha < numLayers - 1; alpha++) // All layers except the input one
   {
      for (int beta = 0; beta < nodes[alpha + 1]; beta++)
      {
         thetaRun = 0.0;
         for (int gamma = 0; gamma < nodes[alpha]; gamma++) // Thetas between current layer and the next
            thetaRun += a[alpha][gamma] * w[alpha][gamma][beta];
         a[alpha + 1][beta] = f(thetaRun);
      }
   } // for (int alpha = 0; alpha < numLayers - 1; alpha++)

   return;
} // void feedForwardRun()

/*
 * This function takes the current weights, and feeds them forward
 * through the network to transform the inputs into the predicted
 * outputs. It uses a threshold function to minimize the effect
 * that nodes with a negative value have. This method is used for
 * training, so it also stores some useful values that will be
 * necessary in gradient descent.
 */
void feedForwardTrain()
{
   for (int alpha = 0; alpha < numLayers - 2; alpha++) // All layers except the first and last
   {
      for (int beta = 0; beta < nodes[alpha + 1]; beta++)
      {
         theta[alpha][beta] = 0.0;
         for (int gamma = 0; gamma < nodes[alpha]; gamma++)
            theta[alpha][beta] += a[alpha][gamma] * w[alpha][gamma][beta];
         a[alpha + 1][beta] = f(theta[alpha][beta]);
      }
   } // for (int alpha = 0; alpha < numLayers - 2; alpha++)

   int alpha = numLayers - 2; // numLayers - 2 is the second to last layer in the network
   for (int beta = 0; beta < nodes[alpha + 1]; beta++)
   {
      theta[alpha][beta] = 0.0;
      for (int gamma = 0; gamma < nodes[alpha]; gamma++)
      {
         theta[alpha][beta] += a[alpha][gamma] * w[alpha][gamma][beta];
      }
      a[alpha + 1][beta] = f(theta[alpha][beta]);
      psi[alpha][beta] = (T[beta] - a[alpha + 1][beta]) * fDer(theta[alpha][beta]);
   } // for (int beta = 0; beta < nodes[alpha + 1]; beta++)

   return;
} // void feedForwardTrain()

/*
 * This function aims to modify the weight vectors to increase the accuracy
 * of the predicted outcome. It essentially takes the partial derivative of
 * the error with respect to the current weights to increment them in a
 * direction that minimizes the error.
 */
void gradientDescent()
{
   double Omega;
   for (int alpha = numLayers - 2; alpha > 1; alpha--)
   {
      for (int beta = 0; beta < nodes[alpha]; beta++)
      {
         Omega = 0.0;
         for (int gamma = 0; gamma < nodes[alpha + 1]; gamma++)
         {
            Omega += psi[alpha][gamma] * w[alpha][beta][gamma];
            w[alpha][beta][gamma] += lambda * (a[alpha][beta] * psi[alpha][gamma]);
         }
         psi[alpha - 1][beta] = Omega * fDer(theta[alpha - 1][beta]); // Update the psi's used for the previous layer
      }
   }

   int alpha = 1; // Special extension of the previous loop to update the first layer weights
   for (int beta = 0; beta < nodes[alpha]; beta++)
   {
      Omega = 0.0;
      for (int gamma = 0; gamma < nodes[alpha + 1]; gamma++)
      {
         Omega += psi[alpha][gamma] * w[alpha][beta][gamma];
         w[alpha][beta][gamma] += lambda * (a[alpha][beta] * psi[alpha][gamma]);
      }

      for (int m = 0; m < nodes[0]; m++) // Update the first layer updates (hence all the 0's)
      {
         w[0][m][beta] += lambda * (a[0][m] * Omega * fDer(theta[0][beta]));
      }
   }

   return;
} // void gradientDescent()

/*
 * Runs through each test case once, printing out the expected vs actual results
 * for each, while also computing the error to print out afterwards.
 */
void run()
{
   std::cout << "Results: " << std::endl << std::endl;
   double err = 0.0;

   for (int c = 0; c < numCases; c++)
   {
      assignCase(c);
      feedForwardRun();

      std::cout << "Test Case " << c + 1 << ": "; //Adding 1 to 1-index the printed results instead of 0-index
      for (int I = 0; I < nodes[numLayers - 1]; I++)
      {
         err += 0.5f * (T[I] - a[numLayers - 1][I]) * (T[I] - a[numLayers - 1][I]);
         std::cout << std::fixed << std::setprecision(17) << a[numLayers - 1][I] << "  ";
      }
      std::cout << std::endl;
   }
   std::cout << "Average Error: " << std::fixed << std::setprecision(5) << err / (double) numCases << std::endl;

   return;
} // void run()

/*
 * Trains the neural network by iterating over the feed-forward and gradient descent
 * functions thousands of times for each test case to minimize the error. Gives some
 * supporting details throughout the process for user consumption (total error messages
 * and termination reason). Calls the run() method at the end to spit out the final
 * results.
 */
void train()
{
   /*
    * Get the initial error by running the feedForward method once over each test case
    */
   double err = 0.0;
   for (int c = 0; c < numCases; c++)
   {
      assignCase(c);
      feedForwardRun();

      for (int I = 0; I < nodes[numLayers - 1]; I++)
      {
         err += 0.5f * (T[I] - a[numLayers - 1][I]) * (T[I] - a[numLayers - 1][I]);
      }
   } // for (int c = 0; c < numCases; c++)

   int iter;
   auto start = std::chrono::high_resolution_clock::now();

   for (iter = 0; iter < iterations && err / (double) numCases > error; iter++)
   {
      err = 0.0;
      for (int c = 0; c < numCases; c++)
      {
         assignCase(c);
         feedForwardTrain();
         gradientDescent();

         for (int I = 0; I < nodes[numLayers - 1]; I++)
            err += 0.5f * (T[I] - a[numLayers - 1][I]) * (T[I] - a[numLayers - 1][I]);
      }

      if (iter % keepAlive == 0)
         std::cout << "Iterations: " << iter << " | Average Error: " << err / (double) numCases << std::endl;
   } // for (int iter = 0; iter < iterations; iter++)

   auto stop = std::chrono::high_resolution_clock::now();
   auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

   std::cout << std::endl;

   std::cout << "Termination Reason(s):" << std::endl;
   if (iter == iterations)
      std::cout << "Max iterations reached" << std::endl;
   if (err / (double) numCases <  error)
      std::cout << "Minimum error threshold reached" << std::endl;

   std::cout << std::endl << "-----------------------------------" << std::endl << std::endl;

   run();
   std::cout << "Total Iterations: " << iter << std::endl;
   std::cout << "Train Time: " << duration.count() << " microseconds" << std::endl;

   std::cout << std::endl << "-----------------------------------" << std::endl << std::endl;

   if (saveLocation.empty())
      std::cout << "Weights are not being saved" << std::endl;
   else
   {
      saveWeights();
      std::cout << "Weights have been saved to " << saveLocation << std::endl;
   }
   std::cout << std::endl;

   return;
} // void train()

/*
 * Simple step-by-step process that determines the code flowchart
 * and is in charge of running everything.
 */
int main(int argc, char** argv)
{
   if (argc == 2) // If a specific config file is given
      setConfig(argv[1]);
   else // Default config file
      setConfig("config0.txt");
   echoConfig();
   allocate();
   populate();
   printCases();

   if (mode == 0)
      train();
   else
      run();

   return 0;
} // int main()
