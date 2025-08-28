#include<bits/stdc++.h>
using namespace std;

struct Delivery {
    int village;
    int d, p, o; // packets dropped
};

struct Trip {
    vector<Delivery> stops;   // villages visited in order
    double distance;          // cached route length
    double value;             // cached total value
    double weight;            // total carried weight
};

struct Helicopter {
    vector<Trip> trips;       // all trips (some may be empty)
    double total_distance;    // sum of trip distances
};

struct State {
    vector<Helicopter> helis; 
    vector<array<int,3>> delivered; // per-village totals (d,p,o)
    double objective;         // cached objective value
};

int main(){

    

}