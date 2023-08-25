#include<iostream>

int main (int argc, char *argv[]) {

    srand((unsigned) time(NULL));

    double old_mean = 0;
    double n_rewards = 0;


    for (int i = 0; i<10; i++) {
        int reward = rand() % 10;

        // Print the random number
        std::cout<<"RANDOM"<<reward<<std::endl;

        old_mean = (old_mean*n_rewards/(n_rewards+1.0))+(reward/(n_rewards+1.0));
        n_rewards++;

        std::cout<<"MEDIA"<<old_mean<<std::endl;
    }
}