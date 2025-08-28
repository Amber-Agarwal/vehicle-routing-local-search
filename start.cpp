#include<bits/stdc++.h>
using namespace std;

struct Village {
    double x;
    double y;
    int n;
};

struct Helicopter_info {
    int home_city;
    double w_cap,dcap,F,alpha;
};
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

int main() {
    double time;
    cin >> time;

    double dmax;
    cin >> dmax;

    double w_d, v_d, w_p, v_p, w_o, v_o;
    cin >> w_d >> v_d >> w_p >> v_p >> w_o >> v_o;

    int C;
    cin >> C;
    vector<pair<double, double>> cities(C);
    for (int i = 0; i < C; ++i) {
        double x, y;
        cin >> x >> y;
        cities[i] = {x, y};
    }

    int V;
    cin >> V;
    vector<Village> villages(V);
    for (int i = 0; i < V; ++i) {
        double x, y;
        int n;
        cin >> x >> y >> n;
        villages[i] = {x, y, n};
    }

    int H;
    cin >> H;
    vector<Helicopter_info> helis(H);
    for (int i = 0; i < H; ++i) {
        int home_city;
        double wcap, dcap, F, alpha;
        cin >> home_city >> wcap >> dcap >> F >> alpha;
        helis[i] = {home_city, wcap, dcap, F, alpha};
    }
    

}