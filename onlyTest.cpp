#include<random>
#include<fstream>
#include<algorithm>
#include<string>
#include<cstdlib>
#include<iostream>

void compute_clause_output_test(
    int &max_literals,
    int &max_clause,
    bool *literal_values,
    bool *clause_output,
    bool **temp_is_included,
    bool *useful_clause
){  
    int i, j; 
    for(i = 0; i<max_clause; i++){
        if(useful_clause[i]){
            clause_output[i] = 1;
            for(j = 0; j<max_literals; j++){
                clause_output[i] = ((!temp_is_included[i][j]) | literal_values[j]) & clause_output[i];
                // std::cout<<clause_output[i]<<" ";
            }
            // std::cout<<std::endl;
        }
        else clause_output[i] = 0;
    }
}

bool predict_class_test(
    int &max_clause,
    bool *clause_type,
    bool *clause_output
){
    int votes = 0, i;
    for(i = 0; i<max_clause; i++){
        if(clause_output[i]){
            if(clause_type[i] == 1){
                votes++;
            }
            else{
                votes--;
            }
        }
    }

    return (votes>=0);
}

void load_arrays(
    int &max_clause,
    int &max_literals,
    bool **temp_is_included,
    bool *clause_type,
    bool *useful_clause
){
    int temp1 = max_clause/2;

    for(int i = 0; i<temp1; i++){
        clause_type[i] = 0;
    }
    for(int i = temp1; i<max_clause; i++){
        clause_type[i] = 1;
    }

    std::ifstream infile("learned_clauses.txt");
    if (!infile) {
        std::cerr << "Error: Could not open clauses.txt" << std::endl;
        return;
    }

    int i = 0;
    std::string line;
    while (std::getline(infile, line)) {
        useful_clause[i] = false;
        for (int j = 0; j < max_literals && j < line.size(); j++) {
            temp_is_included[i][j] = (line[j] == '1');
            if (temp_is_included[i][j]) {
                useful_clause[i] = true;
            }
        }
        // std::cout<<useful_clause[i]<<" "<<clause_type[i]<<std::endl;
        i++;
        if(i>=max_clause) break;
    }

    infile.close();
}

int main() {
    int max_literals = 234;
    int max_clause = 40;
    bool clause_type[max_clause];
    bool clause_output[max_clause];
    bool literal_values[max_literals];
    bool **temp_is_included = new bool*[max_clause];
    for(int i = 0; i<max_clause; i++){
        temp_is_included[i] = new bool[max_literals];
    }
    bool useful_clause[max_clause];
    load_arrays(max_clause, max_literals, temp_is_included, clause_type, useful_clause);
    std::ifstream test_file("test_literals.txt");
    if (test_file) {
        std::string line;
        while (std::getline(test_file, line)) {
            
            for (int i = 0; i < max_literals && i < line.size(); i++) {
                literal_values[i] = (line[i] == '1');
            }
            compute_clause_output_test(max_literals, max_clause, literal_values, clause_output, temp_is_included, useful_clause);
            bool predicted_class = predict_class_test(max_clause, clause_type, clause_output);

            // for(int i = 0; i<max_clause; i++){
            //     std::cout<<clause_output[i]<<" ";
            // }
            // std::cout<<std::endl;
            std::cout << predicted_class << " " << line[max_literals] << std::endl;
        }
        test_file.close();
    } else {
        std::cerr << "Error: Could not open test_literals.txt" << std::endl;
    }
    return 0;
}