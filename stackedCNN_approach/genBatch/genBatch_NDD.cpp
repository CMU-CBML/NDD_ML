#include <iostream>
#include <fstream>
#include <string>

void generateJobFile(int jobIndex) {
    // Define the base content of the job file
    std::string jobName = "ND" + std::to_string(jobIndex);
    std::string fileName = jobName + ".job";
    
    std::ofstream jobFile(fileName);
    
    if (jobFile.is_open()) {
        jobFile << "#!/bin/sh\n";
        jobFile << "#SBATCH -p RM-shared\n";
        jobFile << "#SBATCH --ntasks-per-node=16\n";
        jobFile << "#SBATCH --job-name=" << jobName << "\n";
        jobFile << "#SBATCH --output=" << jobName << ".out\n";
        jobFile << "#SBATCH --time=60:00:00\n";
        jobFile << "#SBATCH --mail-type=ALL\n";
        jobFile << "#SBATCH --mail-user=kuanrenq@andrew.cmu.edu\n";
        jobFile << "mkdir ../io2D_" << jobName << "/\n";
        jobFile << "mkdir ../io2D_" << jobName << "/outputs/\n";
        jobFile << "/ocean/projects/eng170006p/ussqww/petsc/arch-linux-c-opt/bin/mpiexec -np 16 ./2DNG 1 350000 ../io2D_" << jobName << "/\n";
        jobFile << "#-- exit\n";
        jobFile << "#\n";
        jobFile.close();
        std::cout << "File " << fileName << " created successfully!" << std::endl;
    } else {
        std::cerr << "Unable to open file " << fileName << std::endl;
    }
}

void generateSubmitScript(int startIndex, int endIndex) {
    std::ofstream submitFile("submit_jobs.sh");
    
    if (submitFile.is_open()) {
        submitFile << "#!/bin/sh\n\n";
        for (int i = startIndex; i <= endIndex; ++i) {
            submitFile << "sbatch ND" << i << ".job\n";
        }
        submitFile.close();
        std::cout << "submit_jobs.sh created successfully!" << std::endl;
    } else {
        std::cerr << "Unable to open submit_jobs.sh file" << std::endl;
    }
}

int main() {
    int startIndex = 61; // Change this to your desired start index
    int endIndex = 100;   // Change this to your desired end index
    
    // Generate job files
    for (int i = startIndex; i <= endIndex; ++i) {
        generateJobFile(i);
    }
    
    // Generate the submit script
    generateSubmitScript(startIndex, endIndex);
    
    return 0;
}
