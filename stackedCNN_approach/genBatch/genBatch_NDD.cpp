#include <iostream>
#include <fstream>
#include <string>
#include <vector>

void generateJobFile(int jobIndex, int neuronCount, const std::vector<int>& tasksPerNodeVector) {
    // Directly access the tasksPerNode value from the vector
    int tasksPerNode = tasksPerNodeVector[neuronCount - 1];  // neuronCount is 1-based, vector index is 0-based

    // Define the job name and file name to include the number of neurons and switch their position with jobIndex
    std::string jobName = std::to_string(neuronCount) + "_" + std::to_string(jobIndex);  // Switched order
    std::string fileName = jobName + ".job";
    
    std::ofstream jobFile(fileName);
    
    if (jobFile.is_open()) {
        jobFile << "#!/bin/sh\n";
        jobFile << "#SBATCH -p RM-shared\n";
        jobFile << "#SBATCH --ntasks-per-node=" << tasksPerNode << "\n";
        jobFile << "#SBATCH --job-name=" << jobName << "\n";  // Job name includes neuron count first
        jobFile << "#SBATCH --output=" << jobName << ".out\n";  // Output file includes neuron count first
        jobFile << "#SBATCH --time=60:00:00\n";
        jobFile << "#SBATCH --mail-type=ALL\n";
        jobFile << "#SBATCH --mail-user=kuanrenq@andrew.cmu.edu\n";
        jobFile << "mkdir ../io2D_" << jobName << "/\n";
        jobFile << "mkdir ../io2D_" << jobName << "/outputs/\n";
        jobFile << "/ocean/projects/eng170006p/ussqww/petsc/arch-linux-c-opt/bin/mpiexec -np " 
                << tasksPerNode << " ./2DNG " << neuronCount << " 350000 ../io2D_" << jobName << "/\n";
        jobFile << "#-- exit\n";
        jobFile << "#\n";
        jobFile.close();
        std::cout << "File " << fileName << " created successfully!" << std::endl;
    } else {
        std::cerr << "Unable to open file " << fileName << std::endl;
    }
}

void generateSubmitScript(int startIndex, int endIndex, const std::vector<int>& neuronCounts) {
    std::ofstream submitFile("submit_jobs.sh");
    
    if (submitFile.is_open()) {
        submitFile << "#!/bin/sh\n\n";
        for (int neuronCount : neuronCounts) {
            for (int i = startIndex; i <= endIndex; ++i) {
                submitFile << "sbatch " << neuronCount << "_" << i << ".job\n";  // Switched order in the submit script
            }
        }
        submitFile.close();
        std::cout << "submit_jobs.sh created successfully!" << std::endl;
    } else {
        std::cerr << "Unable to open submit_jobs.sh file" << std::endl;
    }
}

int main() {
    int startIndex = 0;  // Change this to your desired start index
    int endIndex = 2;    // Change this to your desired end index

    // List of neuron counts to generate jobs for
    std::vector<int> neuronCounts = {2, 3, 4, 5, 6};  // Adjust this based on neuron cases (example values)

    // Vector holding tasks per node for different neuron counts
    std::vector<int> tasksPerNodeVector = {16, 24, 32, 40, 64, 64};  // Example: 16 for 1 neuron, 24 for 2, etc.

    // Generate job files for each neuron count
    for (int neuronCount : neuronCounts) {
        for (int i = startIndex; i <= endIndex; ++i) {
            generateJobFile(i, neuronCount, tasksPerNodeVector);
        }
    }
    
    // Generate the submit script for all neuron counts
    generateSubmitScript(startIndex, endIndex, neuronCounts);
    
    return 0;
}
