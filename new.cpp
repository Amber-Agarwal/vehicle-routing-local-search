#include <iostream>
#include <vector>
#include <cmath>
#include <array>
#include <algorithm>
#include <random>
#include <numeric>
#include <iomanip>
#include <fstream>
 using namespace std ;
// ===================================================================
// DATA STRUCTURES & GLOBAL VARIABLES
// ===================================================================

struct Village {
    double x;
    double y;
    int n;
};

struct Helicopter_info {
    int home_city;
    double w_cap, dcap, F, alpha;
};

struct Delivery {
    int village;
    int d, p, o; // packets dropped
};

struct Trip {
    std::vector<Delivery> stops;   // villages visited in order
    double distance = 0.0;        // cached route length
    double value = 0.0;           // cached total value
    double weight = 0.0;          // total carried weight
};

struct Helicopter {
    std::vector<Trip> trips;       // all trips (some may be empty)
    double total_distance = 0.0;  // sum of trip distances
};



// Global variables for problem parameters
double DMax;
double w_d, v_d, w_p, v_p, w_o, v_o;
std::vector<std::pair<double, double>> cities;
std::vector<Village> villages;
std::vector<Helicopter_info> helis;
std::vector<std::vector<double>> dist_matrix;
int C, V, H;

struct State {
    std::vector<Helicopter> helis;
    std::vector<std::array<int,3>> delivered; // per-village totals (d,p,o)
    double objective = 0.0;                   // cached objective value
};
enum Destroy_funcs {
    random_stop,
    route_remove,
    shaw,
    worst_values_destroyed,
    perishable_punished
};

enum Repair_funcs{
    greedy_insert,
    // regret2_insert,
    cluster_build,
    random_insert,
    new_insert
};

struct ALNSData {
    std::vector<Destroy_funcs> destroy_names;
    std::vector<Repair_funcs> repair_names;
    std::vector<double> weightD, scoreD, usesD;
    std::vector<double> weightR, scoreR, usesR;
    double r_best = 33.0, r_good = 9.0, r_accepted = 3.0;
    double rho = 0.1;
    int update_window = 50;
    double T0;
    double T;
    std::mt19937 rng;
};

struct RemovedDelivery {
    int v;
    int d,p,o;
};
struct Stop_tracking {int h,t,s;};

struct Pool {
    std::vector<RemovedDelivery> removed;
};

// Forward Declarations
double euclidean_dist(std::pair<double, double> p1, std::pair<double, double> p2);
double calculate_trip_distance(int heli_id, const Trip& trip);
// void fill_trip_data(int heli_id, Trip& trip);
double calculate_trip_weight(const Trip& trip);
void repair_greedy_insert(State &s, Pool &pool, std::mt19937 &rng);
void repair_random_insert(State &s, Pool &pool, std::mt19937 &rng);
void repair_cluster_build(State &s, Pool &pool, std::mt19937 &rng);
void repair_regret2_insert(State &s, Pool &pool, std::mt19937 &rng);
Pool destroy_random_stop(State & s, int num_remove, std::mt19937 & rng);
Pool destroy_route_remove(State & s, int num_remove, std::mt19937 & rng);
Pool destroy_shaw(State & s, int num_remove, std::mt19937 & rng);
Pool destroy_worst_value(State & s, int num_remove, std::mt19937 & rng);
Pool destroy_perishable_aware(State & s, int num_remove, std::mt19937 & rng);
double obj_func(State& state);
State generate_initial_state();


// ===================================================================
// HELPER & UTILITY FUNCTIONS
// ===================================================================

double euclidean_dist(std::pair<double, double> p1, std::pair<double, double> p2) {
    return sqrt(pow(p1.first - p2.first, 2) + pow(p1.second - p2.second, 2));
}

void precompute_distances() {
    int total_nodes = C + V;
    dist_matrix.assign(total_nodes, std::vector<double>(total_nodes, 0.0));
    std::vector<std::pair<double, double>> points;
    points.insert(points.end(), cities.begin(), cities.end());
    for(const auto& v : villages) {
        points.push_back({v.x, v.y});
    }
    for (int i = 0; i < total_nodes; ++i) {
        for (int j = i; j < total_nodes; ++j) {
            double d = euclidean_dist(points[i], points[j]);
            dist_matrix[i][j] = d;
            dist_matrix[j][i] = d;
        }
    }
}

double calculate_trip_weight(const Trip& trip) {
    if (trip.stops.empty()) return 0.0;
    double total_weight = 0.0;
    for (const Delivery& delivery : trip.stops) {
        total_weight += delivery.d * w_d + delivery.p * w_p + delivery.o * w_o;
    }
    return total_weight;
}

double calculate_trip_distance(int heli_id, const Trip& trip) {
    if (trip.stops.empty()) return 0.0;
    double total_dist = 0.0;
    int current_node = helis[heli_id].home_city - 1;
    for (const Delivery& delivery : trip.stops) {
        int next_node = C + delivery.village - 1;
        total_dist += dist_matrix[current_node][next_node];
        current_node = next_node;
    }
    total_dist += dist_matrix[current_node][helis[heli_id].home_city - 1];
    return total_dist;
} 
// IT ALREADY ASSUMES THAT THE PATH IS STARTING FROM A CITY

double calculate_trip_value(const Trip& trip,State& s) {
    double trip_value = 0.0;
    for (const Delivery& delivery : trip.stops) {
        trip_value += delivery.d * v_d + delivery.p * v_p + delivery.o * v_o;
        s.delivered[delivery.village-1][0] += delivery.d;
        s.delivered[delivery.village-1][1] += delivery.p;
        s.delivered[delivery.village-1][2] += delivery.o;
    }
    return trip_value;
}
// IT DOESNT CONSIDER IF GREATER THAN 9K INVOLVED

void fill_trip_data(int heli_id, Trip& trip,State& s) {
    trip.distance = calculate_trip_distance(heli_id, trip);
    trip.weight = calculate_trip_weight(trip);
    trip.value = calculate_trip_value(trip,s);
}


int adaptive_picking(const std::vector<double>& weights, std::mt19937& rng){
    double sum = 0;
    for(auto w: weights) sum += w;
    if (sum <= 1e-9) return std::uniform_int_distribution<int>(0, weights.size() - 1)(rng);
    std::uniform_real_distribution<double> dist(0.0, sum);
    double r = dist(rng);
    double cumulative = 0;
    for(int i=0; i < (int)weights.size(); i++){
        cumulative += weights[i];
        if(r <= cumulative) return i;
    }
    return (int)weights.size() - 1;
}

// ===================================================================
// DESTROY OPERATORS
// ===================================================================

void remove_stop_from_trip(State& s, Pool& pool, const Stop_tracking& ref) {
    Trip &trip = s.helis[ref.h].trips[ref.t];
    Delivery del = trip.stops[ref.s];

    pool.removed.push_back({del.village, del.d, del.p, del.o});
    s.delivered[del.village - 1][0] -= del.d;
    s.delivered[del.village - 1][1] -= del.p;
    s.delivered[del.village - 1][2] -= del.o;
    
    int prev_node = (ref.s == 0) ? helis[ref.h].home_city - 1 : C + trip.stops[ref.s - 1].village - 1;
    int current_node = C + del.village - 1;
    int next_node = (ref.s == (int)trip.stops.size() - 1) ? helis[ref.h].home_city - 1 : C + trip.stops[ref.s + 1].village - 1;

    double dist_change = dist_matrix[prev_node][next_node] - (dist_matrix[prev_node][current_node] + dist_matrix[current_node][next_node]);
    double del_value = del.d * v_d + del.p * v_p + del.o * v_o;
    double cost_change = helis[ref.h].alpha * dist_change;
    
    s.objective -= (del_value - cost_change);

    trip.distance += dist_change;
    s.helis[ref.h].total_distance += dist_change;
    trip.value -= del_value;
    trip.weight -= (del.d * w_d + del.p * w_p + del.o * w_o);

    trip.stops.erase(trip.stops.begin() + ref.s);
    if (trip.stops.empty()) {
        s.objective += helis[ref.h].F; // Add back the cost, since cost is subtracted from objective
    }
}

void perform_sorted_removals(State& s, Pool& pool, std::vector<Stop_tracking>& to_remove_refs) {
    sort(to_remove_refs.begin(), to_remove_refs.end(), [](const auto& a, const auto& b){
        if (a.h != b.h) return a.h > b.h;
        if (a.t != b.t) return a.t > b.t;
        return a.s > b.s;
    });

    for(const auto& ref : to_remove_refs) {
        remove_stop_from_trip(s, pool, ref);
    }
}

Pool destroy_random_stop(State & s, int num_remove, std::mt19937 & rng) {
    Pool pool;
    std::vector<Stop_tracking> candidates;
    for(int h=0; h<s.helis.size(); ++h) for(int t=0; t<s.helis[h].trips.size(); ++t) for(int st=0; st<s.helis[h].trips[t].stops.size(); ++st) candidates.push_back({h,t,st});
    if(candidates.empty()) return pool;
    
    shuffle(candidates.begin(), candidates.end(), rng);
    int num_to_remove = std::min(num_remove, (int)candidates.size());

    std::vector<Stop_tracking> to_remove_refs;
    for(int i=0; i<num_to_remove; ++i) to_remove_refs.push_back(candidates[i]);
    
    perform_sorted_removals(s, pool, to_remove_refs);
    return pool;
}

Pool destroy_route_remove(State & s, int num_remove, std::mt19937 & rng) {
    Pool pool;
    std::vector<std::pair<int,int>> candidates;
    for (int h = 0; h < (int)s.helis.size(); h++) for (int t = 0; t < (int)s.helis[h].trips.size(); t++) if (!s.helis[h].trips[t].stops.empty()) candidates.push_back({h, t});
    if (candidates.empty()) return pool;

    shuffle(candidates.begin(), candidates.end(), rng);
    int remove_count = std::min(num_remove, (int)candidates.size());

    for (int k = 0; k < remove_count; k++) {
        auto [h, t] = candidates[k];
        Trip &trip = s.helis[h].trips[t];
        
        if (trip.value > 1e-9 || trip.distance > 1e-9) {
            s.objective -= (trip.value - (helis[h].alpha * trip.distance + helis[h].F));
        }
        
        for (const Delivery &del : trip.stops) {
            pool.removed.push_back({del.village, del.d, del.p, del.o});
            s.delivered[del.village - 1][0] -= del.d;
            s.delivered[del.village - 1][1] -= del.p;
            s.delivered[del.village - 1][2] -= del.o;
        }
        
        s.helis[h].total_distance -= trip.distance;
        trip = Trip();
    }
    return pool;
}

Pool destroy_shaw(State &s, int num_remove, std::mt19937 &rng) {
    Pool pool;
    std::vector<Stop_tracking> candidates;
    for(int h=0; h<s.helis.size(); ++h) for(int t=0; t<s.helis[h].trips.size(); ++t) for(int st=0; st<s.helis[h].trips[t].stops.size(); ++st) candidates.push_back({h,t,st});
    if(candidates.empty()) return pool;

    std::uniform_int_distribution<int> pick(0, candidates.size()-1);
    Stop_tracking seed_ref = candidates[pick(rng)];
    int seed_village = s.helis[seed_ref.h].trips[seed_ref.t].stops[seed_ref.s].village;
    
    std::vector<std::pair<double, Stop_tracking>> scored;
    for(const auto& ref : candidates) {
        int v_idx = s.helis[ref.h].trips[ref.t].stops[ref.s].village;
        double d = dist_matrix[C+seed_village-1][C+v_idx-1];
        scored.push_back({-d, ref});
    }

    sort(scored.begin(), scored.end(), [](auto &a, auto &b) { return a.first > b.first; });
    
    int num_to_remove = std::min(num_remove, (int)scored.size());
    std::vector<Stop_tracking> to_remove_refs;
    for(int i=0; i<num_to_remove; ++i) to_remove_refs.push_back(scored[i].second);
    
    perform_sorted_removals(s, pool, to_remove_refs);
    return pool;
}

Pool destroy_worst_value(State & s, int num_remove, std::mt19937 & rng) {
    Pool pool;
    std::vector<std::pair<double, Stop_tracking>> scored;

    for (int h = 0; h < (int)s.helis.size(); h++) {
        for (int t = 0; t < (int)s.helis[h].trips.size(); t++) {
            const Trip &trip = s.helis[h].trips[t];
            for (int st = 0; st < (int)trip.stops.size(); st++) {
                const Delivery &del = trip.stops[st];
                double val = del.d * v_d + del.p * v_p + del.o * v_o;
                
                int prev_node = (st == 0) ? helis[h].home_city - 1 : C + trip.stops[st - 1].village - 1;
                int current_node = C + del.village - 1;
                int next_node = (st == (int)trip.stops.size() - 1) ? helis[h].home_city - 1 : C + trip.stops[st + 1].village - 1;

                double dist_contribution = dist_matrix[prev_node][current_node] + dist_matrix[current_node][next_node] - dist_matrix[prev_node][next_node];
                double score = val / std::max(1e-9, dist_contribution);
                scored.push_back({score, {h,t,st}});
            }
        }
    }
    if (scored.empty()) return pool;
    
    sort(scored.begin(), scored.end(), [](auto &a, auto &b) { return a.first < b.first; });

    int num_to_remove = std::min(num_remove, (int)scored.size());
    std::vector<Stop_tracking> to_remove_refs;
    for(int i=0; i<num_to_remove; ++i) to_remove_refs.push_back(scored[i].second);
    
    perform_sorted_removals(s, pool, to_remove_refs);
    return pool;
}

Pool destroy_perishable_aware(State & s, int num_remove, std::mt19937 & rng) {
    Pool pool;
    std::vector<std::pair<double, Stop_tracking>> scored;
    std::vector<int> food_limit(V);
    for (int v = 0; v < V; v++) food_limit[v] = 9 * villages[v].n;

    for (int h = 0; h < (int)s.helis.size(); h++) {
        for (int t = 0; t < (int)s.helis[h].trips.size(); t++) {
            const Trip &trip = s.helis[h].trips[t];
            for (int st = 0; st < (int)trip.stops.size(); st++) {
                const Delivery &del = trip.stops[st];
                int v = del.village - 1;
                double total_food = del.d + del.p;
                if (total_food < 1e-9) continue;

                double frac_dry = (double)del.d / total_food;
                int current_total_food = s.delivered[v][0] + s.delivered[v][1];
                int overuse = std::max(0, current_total_food - food_limit[v]);
                
                double score = frac_dry + overuse * 0.1; 
                scored.push_back({score, {h,t,st}});
            }
        }
    }
    if (scored.empty()) return pool;

    sort(scored.begin(), scored.end(), [](auto &a, auto &b){ return a.first > b.first; });

    int num_to_remove = std::min(num_remove, (int)scored.size());
    std::vector<Stop_tracking> to_remove_refs;
    for(int i=0; i<num_to_remove; ++i) to_remove_refs.push_back(scored[i].second);
    
    perform_sorted_removals(s, pool, to_remove_refs);
    return pool;
}

// This is a local search operator that modifies the state directly.
// It returns 'true' if it successfully made a change.
bool change_quantity(State& s, std::mt19937& rng) {
    // Collect all possible stops that can be modified
    std::vector<Stop_tracking> candidates;
    for (int h = 0; h < s.helis.size(); ++h) {
        for (int t = 0; t < s.helis[h].trips.size(); ++t) {
            for (int stop_idx = 0; stop_idx < s.helis[h].trips[t].stops.size(); ++stop_idx) {
                candidates.push_back({h, t, stop_idx});
            }
        }
    }

    if (candidates.empty()) {
        return false;
    }

    // Pick a random stop to try and improve
    std::uniform_int_distribution<int> dist(0, candidates.size() - 1);
    Stop_tracking& ref = candidates[dist(rng)];
    
    Trip& trip = s.helis[ref.h].trips[ref.t];
    Delivery& del = trip.stops[ref.s];
    
    // Calculate spare weight on this trip
    double spare_weight = helis[ref.h].w_cap - trip.weight;
    if (spare_weight < 1e-9) {
        return false; // No room for improvement
    }

    // Find out how many more packets this village needs
    int v_idx = del.village - 1;
    int food_limit = 9 * villages[v_idx].n;
    int other_limit = 1 * villages[v_idx].n;
    int p_needed = std::max(0, food_limit - (s.delivered[v_idx][0] + s.delivered[v_idx][1]));
    int o_needed = std::max(0, other_limit - s.delivered[v_idx][2]);

    double initial_objective = s.objective;

    // Try to add valuable 'p' packets first
    int p_to_add = std::min(p_needed, (int)(spare_weight / w_p));
    if (p_to_add > 0) {
        del.p += p_to_add;
        double added_weight = p_to_add * w_p;
        double added_value = p_to_add * v_p;

        trip.weight += added_weight;
        trip.value += added_value;
        s.delivered[v_idx][1] += p_to_add;
        s.objective += added_value; // Simple objective update
        spare_weight -= added_weight;
    }

    // Try to add 'o' packets with the remaining capacity
    int o_to_add = std::min(o_needed, (int)(spare_weight / w_o));
    if (o_to_add > 0) {
        del.o += o_to_add;
        double added_weight = o_to_add * w_o;
        double added_value = o_to_add * v_o;
        
        trip.weight += added_weight;
        trip.value += added_value;
        s.delivered[v_idx][2] += o_to_add;
        s.objective += added_value;
    }

    // If we made a change, re-evaluate the true objective for this village
    // to handle food limits correctly. This is safer than complex incremental updates.
    if (p_to_add > 0 || o_to_add > 0) {
        obj_func(s); // Use the full objective function to re-sync the score
        return true;
    }

    return false;
}

// This is a local search operator that tries to swap packet types within a delivery.
// Returns 'true' if it successfully made a change.
bool swap_packet_type(State& s, std::mt19937& rng) {
    // Find a delivery that has at least one perishable ('p') packet to swap out
    std::vector<Stop_tracking> candidates;
    for (int h = 0; h < s.helis.size(); ++h) {
        for (int t = 0; t < s.helis[h].trips.size(); ++t) {
            for (int st = 0; st < s.helis[h].trips[t].stops.size(); ++st) {
                if (s.helis[h].trips[t].stops[st].p > 0) {
                    candidates.push_back({h, t, st});
                }
            }
        }
    }

    if (candidates.empty()) {
        return false;
    }

    // Pick a random delivery to modify
    std::uniform_int_distribution<int> dist(0, candidates.size() - 1);
    Stop_tracking& ref = candidates[dist(rng)];

    Trip& trip = s.helis[ref.h].trips[ref.t];
    Delivery& del = trip.stops[ref.s];
    int v_idx = del.village - 1;

    // Calculate how many 'd' packets we can get for one 'p' packet
    int d_per_p = floor(w_p / w_d);
    if (d_per_p == 0) return false; // 'd' packets are heavier than 'p'

    // Calculate the value change. This move is only useful if the swap is profitable.
    double value_change = (d_per_p * v_d) - v_p;
    if (value_change <= 0) return false; // The swap isn't worth it

    // Check if the village actually needs more food
    int food_limit = 9 * villages[v_idx].n;
    int current_food = s.delivered[v_idx][0] + s.delivered[v_idx][1];
    if (current_food >= food_limit) return false;

    // --- Execute the Swap ---
    // Remove one 'p' packet
    del.p--;
    trip.weight -= w_p;
    s.delivered[v_idx][1]--;

    // Add the equivalent weight in 'd' packets
    del.d += d_per_p;
    trip.weight += d_per_p * w_d;
    s.delivered[v_idx][0] += d_per_p;

    // The trip's total weight might change slightly due to floating point,
    // so it's safest to recalculate it from the stops.
    trip.weight = calculate_trip_weight(trip);

    // Recalculate everything for safety and correctness
    obj_func(s);
    return true;
}
// ===================================================================
// REPAIR OPERATORS
// ===================================================================

void repair_greedy_insert(State &s, Pool &pool, std::mt19937 &rng) {
    shuffle(pool.removed.begin(), pool.removed.end(), rng);
    auto it = pool.removed.begin();
    while (it != pool.removed.end()) {
        Delivery del{it->v, it->d, it->p, it->o};
        double del_value = del.d * v_d + del.p * v_p + del.o * v_o;
        double del_weight = del.d * w_d + del.p * w_p + del.o * w_o;

        double best_gain = -1e18;
        int best_h = -1, best_t = -1, best_pos = -1;
        double best_extra_dist = 0.0;

        for (int h = 0; h < (int)s.helis.size(); h++) {
            for (int t = 0; t < (int)s.helis[h].trips.size(); t++) {
                Trip &trip = s.helis[h].trips[t];
                for (int pos = 0; pos <= (int)trip.stops.size(); pos++) {
                    int prev_node = (pos == 0) ? helis[h].home_city - 1 : C + trip.stops[pos - 1].village - 1;
                    int current_node = C + del.village - 1;
                    int next_node = (pos == (int)trip.stops.size()) ? helis[h].home_city - 1 : C + trip.stops[pos].village - 1;
                    
                    double extra_dist = dist_matrix[prev_node][current_node] + dist_matrix[current_node][next_node] - dist_matrix[prev_node][next_node];
                    if (trip.distance + extra_dist > helis[h].dcap + 1e-9) continue;
                    if (s.helis[h].total_distance + extra_dist > DMax + 1e-9) continue;
                    if (trip.weight + del_weight > helis[h].w_cap + 1e-9) continue;

                    double gain = del_value - helis[h].alpha*extra_dist;
                    if (gain > best_gain) {
                        best_gain = gain; best_h = h; best_t = t; best_pos = pos; best_extra_dist = extra_dist;
                    }
                }
            }
        }

        bool inserted = false;
        if (best_gain > -1e17) {
            Trip &trip = s.helis[best_h].trips[best_t];
            trip.stops.insert(trip.stops.begin() + best_pos, del);
            trip.distance += best_extra_dist; trip.value += del_value; trip.weight += del_weight;
            s.helis[best_h].total_distance += best_extra_dist;
            s.delivered[del.village - 1][0] += del.d; s.delivered[del.village - 1][1] += del.p; s.delivered[del.village - 1][2] += del.o;
            s.objective += best_gain;
            inserted = true;
        } else {
            double best_new_trip_gain = -1e18; int best_h_for_new_trip = -1;
            for (int h = 0; h < (int)s.helis.size(); h++) {
                double trip_dist = dist_matrix[helis[h].home_city-1][C+del.village-1] * 2;
                if (trip_dist <= helis[h].dcap + 1e-9 && del_weight <= helis[h].w_cap + 1e-9 && s.helis[h].total_distance + trip_dist <= DMax + 1e-9) {
                    double new_trip_gain = del_value - helis[h].alpha * trip_dist - helis[h].F;
                    if (new_trip_gain > best_new_trip_gain) {
                        best_new_trip_gain = new_trip_gain; best_h_for_new_trip = h;
                    }
                }
            }
            if (best_h_for_new_trip != -1) {
                int h = best_h_for_new_trip;
                Trip new_trip; new_trip.stops.push_back(del);
                new_trip.weight = del_weight; new_trip.value = del_value; new_trip.distance = dist_matrix[helis[h].home_city-1][C+del.village-1] * 2;
                
                s.helis[h].trips.push_back(new_trip);
                s.helis[h].total_distance += new_trip.distance;
                s.delivered[del.village-1][0] += del.d; s.delivered[del.village-1][1] += del.p; s.delivered[del.village-1][2] += del.o;
                s.objective += best_new_trip_gain;
                inserted = true;
            }
        }

        if (inserted) {
            it = pool.removed.erase(it);
        } else {
            ++it;
        }
    }
}

void repair_random_insert(State &s, Pool &pool, std::mt19937 &rng) {
    shuffle(pool.removed.begin(), pool.removed.end(), rng);
    auto it = pool.removed.begin();
    while(it != pool.removed.end()){
        Delivery del{it->v, it->d, it->p, it->o};
        double del_value = del.d*v_d + del.p*v_p + del.o*v_o;
        double del_weight = del.d*w_d + del.p*w_p + del.o*w_o;
        struct FeasibleInsertion { int h,t,pos; double extra_dist; };
        std::vector<FeasibleInsertion> feasible_spots;

        for(int h=0; h<s.helis.size(); ++h) for(int t=0; t<s.helis[h].trips.size(); ++t) {
            Trip& trip = s.helis[h].trips[t];
            for(int pos=0; pos<=trip.stops.size(); ++pos) {
                int prev_node = (pos == 0) ? helis[h].home_city - 1 : C + trip.stops[pos - 1].village - 1;
                int current_node = C + del.village - 1;
                int next_node = (pos == (int)trip.stops.size()) ? helis[h].home_city - 1 : C + trip.stops[pos].village - 1;
                
                double extra_dist = dist_matrix[prev_node][current_node] + dist_matrix[current_node][next_node] - dist_matrix[prev_node][next_node];
                if (trip.distance+extra_dist > helis[h].dcap+1e-9 || s.helis[h].total_distance+extra_dist > DMax+1e-9 || trip.weight+del_weight > helis[h].w_cap+1e-9) continue;
                feasible_spots.push_back({h,t,pos,extra_dist});
            }
        }
        
        bool inserted = false;
        if (!feasible_spots.empty()) {
            std::uniform_int_distribution<int> dist(0, feasible_spots.size()-1);
            FeasibleInsertion spot = feasible_spots[dist(rng)];
            Trip& trip = s.helis[spot.h].trips[spot.t];
            trip.stops.insert(trip.stops.begin()+spot.pos, del);
            trip.distance += spot.extra_dist; trip.value += del_value; trip.weight += del_weight;
            s.helis[spot.h].total_distance += spot.extra_dist;
            s.delivered[del.village-1][0]+=del.d; s.delivered[del.village-1][1]+=del.p; s.delivered[del.village-1][2]+=del.o;
            s.objective += del_value - helis[spot.h].alpha * spot.extra_dist;
            inserted = true;
        }

        if (inserted) {
            it = pool.removed.erase(it);
        } else {
            ++it;
        }
    }
}

void repair_cluster_build(State &s, Pool &pool, std::mt19937 &rng) {
    std::vector<bool> used(pool.removed.size(), false);
    int remaining = pool.removed.size();

    while (remaining > 0) {
        int seed_idx = -1;
        std::uniform_int_distribution<int> dist(0, remaining - 1);
        int skip = dist(rng);
        for(int i = 0; i < pool.removed.size(); ++i) {
            if (!used[i]) {
                if (skip == 0) {
                    seed_idx = i;
                    break;
                }
                skip--;
            }
        }
        if (seed_idx == -1) break; 
        
        used[seed_idx] = true;
        remaining--;
        RemovedDelivery seed_del = pool.removed[seed_idx];

        int best_h = -1;
        double min_dist_to_seed = 1e18;
        for (int h = 0; h < (int)s.helis.size(); ++h) {
            double d = dist_matrix[helis[h].home_city - 1][C + seed_del.v - 1];
            if (d < min_dist_to_seed) {
                min_dist_to_seed = d;
                best_h = h;
            }
        }
        if (best_h == -1) continue;

        Trip new_trip;
        new_trip.stops.push_back({seed_del.v, seed_del.d, seed_del.p, seed_del.o});
        
        new_trip.weight = seed_del.d * w_d + seed_del.p * w_p + seed_del.o * w_o;
        new_trip.value = seed_del.d * v_d + seed_del.p * v_p + seed_del.o * v_o;
        new_trip.distance = dist_matrix[helis[best_h].home_city-1][C+seed_del.v-1] * 2;

        while (remaining > 0) {
            int last_village_idx = new_trip.stops.back().village - 1;
            int best_next_del_idx = -1;
            double min_next_dist = 1e18;

            for (int i = 0; i < (int)pool.removed.size(); ++i) {
                if (!used[i]) {
                    double d = dist_matrix[C + last_village_idx][C + pool.removed[i].v - 1];
                    if (d < min_next_dist) {
                        min_next_dist = d;
                        best_next_del_idx = i;
                    }
                }
            }

            if (best_next_del_idx == -1) break;

            RemovedDelivery next_del = pool.removed[best_next_del_idx];
            double next_del_weight = next_del.d * w_d + next_del.p * w_p + next_del.o * w_o;
            
            int last_node = C + last_village_idx;
            int next_node = C + next_del.v - 1;
            int home_node = helis[best_h].home_city - 1;
            double extra_dist = dist_matrix[last_node][next_node] + dist_matrix[next_node][home_node] - dist_matrix[last_node][home_node];

            double proposed_weight = new_trip.weight + next_del_weight;
            double proposed_dist = new_trip.distance + extra_dist;

            if (proposed_dist <= helis[best_h].dcap + 1e-9 && proposed_weight <= helis[best_h].w_cap + 1e-9 && s.helis[best_h].total_distance + proposed_dist <= DMax + 1e-9) {
                new_trip.stops.push_back({next_del.v, next_del.d, next_del.p, next_del.o});
                new_trip.weight = proposed_weight;
                new_trip.distance = proposed_dist;
                new_trip.value += next_del.d * v_d + next_del.p * v_p + next_del.o * v_o;
                
                used[best_next_del_idx] = true;
                remaining--;
            } else {
                break;
            }
        }
        
        if (!new_trip.stops.empty()) {
            s.helis[best_h].trips.push_back(new_trip);
            s.helis[best_h].total_distance += new_trip.distance;
            double new_trip_gain = new_trip.value - helis[best_h].alpha * new_trip.distance - helis[best_h].F;
            s.objective += new_trip_gain;
            for (const auto& del : new_trip.stops) {
                s.delivered[del.village - 1][0] += del.d;
                s.delivered[del.village - 1][1] += del.p;
                s.delivered[del.village - 1][2] += del.o;
            }
        }
    }
}

struct InsertionCandidate {
    double gain;
    int h, t, pos;
    double extra_dist;
    bool operator<(const InsertionCandidate& other) const {
        return gain < other.gain;
    }
};

// void repair_regret2_insert(State &s, Pool &pool, std::mt19937 &rng) {
//     while (!pool.removed.empty()) {
//         double max_regret = -1e18;
//         int best_del_idx = -1;
//         InsertionCandidate best_insertion_for_max_regret;

//         for (int i = 0; i < pool.removed.size(); ++i) {
//             RemovedDelivery& del = pool.removed[i];
//             double del_value = del.d * v_d + del.p * v_p + del.o * v_o;
//             double del_weight = del.d * w_d + del.p * w_p + del.o * w_o;

//             std::vector<InsertionCandidate> candidates;

//             for (int h = 0; h < (int)s.helis.size(); h++) {
//                 for (int t = 0; t < (int)s.helis[h].trips.size(); t++) {
//                     Trip &trip = s.helis[h].trips[t];
//                     for (int pos = 0; pos <= (int)trip.stops.size(); pos++) {
//                         int prev_node = (pos == 0) ? helis[h].home_city - 1 : C + trip.stops[pos - 1].village - 1;
//                         int current_node = C + del.v - 1;
//                         int next_node = (pos == (int)trip.stops.size()) ? helis[h].home_city - 1 : C + trip.stops[pos].village - 1;

//                         double extra_dist = dist_matrix[prev_node][current_node] + dist_matrix[current_node][next_node] - dist_matrix[prev_node][next_node];
//                         if (trip.distance + extra_dist > helis[h].dcap + 1e-9) continue;
//                         if (s.helis[h].total_distance + extra_dist > DMax + 1e-9) continue;
//                         if (trip.weight + del_weight > helis[h].w_cap + 1e-9) continue;
                        
//                         candidates.push_back({del_value - helis[h].alpha * extra_dist, h, t, pos, extra_dist});
//                     }
//                 }
//             }
//             if (candidates.empty()) continue;
//             sort(candidates.rbegin(), candidates.rend()); 
//             double regret = (candidates.size() >= 2) ? (candidates[0].gain - candidates[1].gain) : (candidates[0].gain > 0 ? candidates[0].gain : 1e18);
//             if (regret > max_regret) {
//                 max_regret = regret;
//                 best_del_idx = i;
//                 best_insertion_for_max_regret = candidates[0];
//             }
//         }

//         if (best_del_idx == -1) {
//             repair_greedy_insert(s, pool, rng);
//             pool.removed.clear();
//             return;
//         }

//         RemovedDelivery del_to_insert = pool.removed[best_del_idx];
//         InsertionCandidate &best_spot = best_insertion_for_max_regret;
        
//         Trip &trip = s.helis[best_spot.h].trips[best_spot.t];
//         trip.stops.insert(trip.stops.begin() + best_spot.pos, {del_to_insert.v, del_to_insert.d, del_to_insert.p, del_to_insert.o});

//         double del_value = del_to_insert.d*v_d + del_to_insert.p*v_p + del_to_insert.o*v_o;
//         double del_weight = del_to_insert.d*w_d + del_to_insert.p*w_p + del_to_insert.o*w_o;
//         trip.distance += best_spot.extra_dist;
//         trip.value += del_value;
//         trip.weight += del_weight;
//         s.helis[best_spot.h].total_distance += best_spot.extra_dist;
//         s.delivered[del_to_insert.v - 1][0] += del_to_insert.d;
//         s.delivered[del_to_insert.v - 1][1] += del_to_insert.p;
//         s.delivered[del_to_insert.v - 1][2] += del_to_insert.o;
//         s.objective += best_spot.gain;

//         pool.removed.erase(pool.removed.begin() + best_del_idx);
//     }
// }
// #include <algorithm> // For std::min

void repair_and_adjust_quantities(State &s, Pool &pool, std::mt19937 &rng) {
    // Shuffle the pool to process requests in a random order, introducing diversity
    shuffle(pool.removed.begin(), pool.removed.end(), rng);

    auto it = pool.removed.begin();
    while (it != pool.removed.end()) {
        RemovedDelivery request = *it;

        // --- STAGE 1: SEARCH FOR THE BEST POSSIBLE ACTION ---
        double best_gain = -1e18;
        bool is_action_found = false;
        bool is_new_trip_action = false;
        
        // Variables to store the details of the best action
        int best_h = -1, best_t = -1, best_pos = -1;
        Delivery best_adjusted_del;
        double best_action_dist_change = 0.0;

        // --- Part A: Try to insert into an EXISTING trip ---
        for (int h = 0; h < (int)s.helis.size(); ++h) {
            for (int t = 0; t < (int)s.helis[h].trips.size(); ++t) {
                Trip& trip = s.helis[h].trips[t];
                
                // Don't insert into trips that are already empty (let the "new trip" logic handle that)
                if (trip.stops.empty()) continue;

                for (int pos = 0; pos <= (int)trip.stops.size(); ++pos) {
                    // Calculate spare capacity of this trip
                    double spare_weight = helis[h].w_cap - trip.weight;
                    
                    // Create an optimized delivery based on spare capacity and request needs
                    Delivery adjusted_del = {request.v, 0, 0, 0};
                    int p_to_add = std::min(request.p, (int)(spare_weight / w_p));
                    adjusted_del.p = p_to_add;
                    spare_weight -= p_to_add * w_p;
                    int o_to_add = std::min(request.o, (int)(spare_weight / w_o));
                    adjusted_del.o = o_to_add;

                    if (adjusted_del.p == 0 && adjusted_del.o == 0) continue; // Cannot fit anything

                    double adjusted_weight = adjusted_del.p * w_p + adjusted_del.o * w_o;
                    double adjusted_value = adjusted_del.p * v_p + adjusted_del.o * v_o;
                    
                    // Calculate the distance change for this new, adjusted delivery
                    int prev_node = (pos == 0) ? helis[h].home_city - 1 : C + trip.stops[pos-1].village - 1;
                    int current_node = C + adjusted_del.village - 1;
                    int next_node = (pos == trip.stops.size()) ? helis[h].home_city - 1 : C + trip.stops[pos].village - 1;
                    double extra_dist = dist_matrix[prev_node][current_node] + dist_matrix[current_node][next_node] - dist_matrix[prev_node][next_node];

                    // Final feasibility check
                    if (trip.distance + extra_dist <= helis[h].dcap + 1e-9 &&
                        s.helis[h].total_distance + extra_dist <= DMax + 1e-9) {
                        
                        double gain = adjusted_value - helis[h].alpha * extra_dist;
                        if (gain > best_gain) {
                            best_gain = gain;
                            is_action_found = true;
                            is_new_trip_action = false;
                            best_h = h; best_t = t; best_pos = pos;
                            best_adjusted_del = adjusted_del;
                            best_action_dist_change = extra_dist;
                        }
                    }
                }
            }
        }

        // --- Part B: Try to create a NEW trip ---
        for (int h = 0; h < (int)s.helis.size(); ++h) {
            double spare_weight = helis[h].w_cap; // A new trip has full weight capacity
            
            Delivery adjusted_del = {request.v, 0, 0, 0};
            int p_to_add = std::min(request.p, (int)(spare_weight / w_p));
            adjusted_del.p = p_to_add;
            spare_weight -= p_to_add * w_p;
            int o_to_add = std::min(request.o, (int)(spare_weight / w_o));
            adjusted_del.o = o_to_add;

            if (adjusted_del.p == 0 && adjusted_del.o == 0) continue;

            double adjusted_weight = adjusted_del.p * w_p + adjusted_del.o * w_o;
            double adjusted_value = adjusted_del.p * v_p + adjusted_del.o * v_o;
            double trip_dist = dist_matrix[helis[h].home_city-1][C+adjusted_del.village-1] * 2;

            if (trip_dist <= helis[h].dcap + 1e-9 && s.helis[h].total_distance + trip_dist <= DMax + 1e-9) {
                double gain = adjusted_value - helis[h].alpha * trip_dist - helis[h].F;
                if (gain > best_gain) {
                    best_gain = gain;
                    is_action_found = true;
                    is_new_trip_action = true;
                    best_h = h;
                    best_adjusted_del = adjusted_del;
                    best_action_dist_change = trip_dist;
                }
            }
        }
        
        // --- STAGE 2: EXECUTE THE BEST ACTION ---
        if (is_action_found) {
            if (is_new_trip_action) {
                Trip new_trip;
                new_trip.stops.push_back(best_adjusted_del);
                new_trip.weight = best_adjusted_del.p * w_p + best_adjusted_del.o * w_o;
                new_trip.value = best_adjusted_del.p * v_p + best_adjusted_del.o * v_o;
                new_trip.distance = best_action_dist_change;
                s.helis[best_h].trips.push_back(new_trip);
            } else { // Insert into existing trip
                Trip& trip = s.helis[best_h].trips[best_t];
                trip.stops.insert(trip.stops.begin() + best_pos, best_adjusted_del);
                trip.distance += best_action_dist_change;
                trip.value += best_adjusted_del.p * v_p + best_adjusted_del.o * v_o;
                trip.weight += best_adjusted_del.p * w_p + best_adjusted_del.o * w_o;
            }
            
            // Universal state updates
            s.helis[best_h].total_distance += best_action_dist_change;
            s.delivered[request.v-1][1] += best_adjusted_del.p;
            s.delivered[request.v-1][2] += best_adjusted_del.o;
            s.objective += best_gain;

            // --- STAGE 3: UPDATE THE POOL ---
            it->p -= best_adjusted_del.p;
            it->o -= best_adjusted_del.o;
            if (it->p <= 0 && it->o <= 0) {
                it = pool.removed.erase(it); // Request is fully satisfied
            } else {
                ++it; // Request partially satisfied, leave remainder for next iteration
            }
        } else {
            ++it; // No feasible action found for this request, skip it
        }
    }
}

// ===================================================================
// FULL OBJECTIVE FUNCTION (FOR VALIDATION & INITIALIZATION)
// ===================================================================


//IT DOESNT CHECK IF THE HELICOPTER START AND END FROM THE HOME CITY
double obj_func(State& state) {
    double total_value = 0.0;
    double total_cost = 0.0;
    state.delivered.assign(V, std::array<int,3>{0, 0, 0});

    // Recalculate all trip data from scratch to be safe
    for (int h = 0; h < H; h++) {
        for (Trip& trip : state.helis[h].trips) {
            fill_trip_data(h, trip,state);
        }
    }
    
    // Recalculate helicopter distances
    for (int h=0; h<H; ++h) {
        state.helis[h].total_distance = 0;
        for(const auto& trip : state.helis[h].trips) {
            state.helis[h].total_distance += trip.distance;
        }
    }

    // Recalculate delivered totals
    // for (const auto& heli : state.helis) {
    //     for (const auto& trip : heli.trips) {
    //         for (const auto& del : trip.stops) {
    //             state.delivered[del.village - 1][0] += del.d;
    //             state.delivered[del.village - 1][1] += del.p;
    //             state.delivered[del.village - 1][2] += del.o;
    //         }
    //     }
    // }
    
    for (int h = 0; h < H; h++) {
        const Helicopter& heli = state.helis[h];
        if (heli.total_distance > DMax + 1e-9) return -1e18; // Infeasible
        
        for (const Trip& trip : heli.trips) {
            if (trip.stops.empty()) continue;
            if (trip.weight > helis[h].w_cap + 1e-9) return -1e18; // Infeasible
            if (trip.distance > helis[h].dcap + 1e-9) return -1e18; // Infeasible
            
            total_cost += helis[h].F + helis[h].alpha * trip.distance;
        }
    }
    
    for (int v_idx = 0; v_idx < V; v_idx++) {
        int people = villages[v_idx].n;
        int food_limit = 9 * people;
        
        int total_food = state.delivered[v_idx][0] + state.delivered[v_idx][1];
        int useful_food = std::min(total_food, food_limit);
        
        int perishable = state.delivered[v_idx][1];
        if (useful_food <= perishable) {
            total_value += useful_food * v_p;
        } else {
            total_value += perishable * v_p;
            total_value += (useful_food - perishable) * v_d;
        }
        total_value += state.delivered[v_idx][2] * v_o;
    }
    
    state.objective = total_value - total_cost;
    return state.objective;
}

// ===================================================================
// INITIAL STATE GENERATION
// ===================================================================

State generate_initial_state() {
    State state;
    state.helis.resize(H);
    state.delivered.assign(V, {0,0,0});
    for(int h=0; h<H; ++h) state.helis[h].trips.push_back(Trip());

    std::vector<int> village_indices(V);
    std::iota(village_indices.begin(), village_indices.end(), 0);
    
    for(int v_idx : village_indices) {
        int people = villages[v_idx].n;
        int food_needed = people * 9;
        int other_needed = people * 1;
        
        int p_packs = std::min(food_needed, (int)(helis[0].w_cap / w_p)); // Simple allocation
        int o_packs = std::min(other_needed, (int)((helis[0].w_cap - p_packs*w_p) / w_o));
        
        if (p_packs + o_packs == 0) continue;

        Delivery del = {v_idx + 1, 0, p_packs, o_packs};
        double del_value = del.d*v_d + del.p*v_p + del.o*v_o;
        double del_weight = del.d*w_d + del.p*w_p + del.o*w_o;
        
        double best_gain = -1e18;
        int best_h = -1, best_t = -1, best_pos = -1;
        double best_extra_dist = 0.0;
        bool is_new_trip = false;

        // Try inserting into existing trips
        for (int h = 0; h < H; h++) {
            for (int t = 0; t < state.helis[h].trips.size(); t++) {
                Trip &trip = state.helis[h].trips[t];
                 if (trip.stops.empty()) continue;
                for (int pos = 0; pos <= trip.stops.size(); pos++) {
                    int prev_node = (pos == 0) ? helis[h].home_city - 1 : C + trip.stops[pos - 1].village - 1;
                    int current_node = C + del.village - 1;
                    int next_node = (pos == (int)trip.stops.size()) ? helis[h].home_city - 1 : C + trip.stops[pos].village - 1;
                    double extra_dist = dist_matrix[prev_node][current_node] + dist_matrix[current_node][next_node] - dist_matrix[prev_node][next_node];
                    
                    if (trip.distance+extra_dist <= helis[h].dcap && state.helis[h].total_distance+extra_dist <= DMax && trip.weight+del_weight <= helis[h].w_cap) {
                        double gain = del_value - helis[h].alpha * extra_dist;
                        if (gain > best_gain) {
                            best_gain = gain; best_h = h; best_t = t; best_pos = pos; best_extra_dist = extra_dist; is_new_trip=false;
                        }
                    }
                }
            }
        }
        
        // Try creating a new trip
        for (int h=0; h<H; ++h) {
            double trip_dist = dist_matrix[helis[h].home_city-1][C+del.village-1] * 2;
            if (trip_dist <= helis[h].dcap && del_weight <= helis[h].w_cap && state.helis[h].total_distance + trip_dist <= DMax) {
                double gain = del_value - helis[h].alpha*trip_dist - helis[h].F;
                if (gain > best_gain) {
                    best_gain = gain; best_h = h; best_t = -1; best_extra_dist = trip_dist; is_new_trip = true;
                }
            }
        }
        
        if (best_h != -1) {
            if (is_new_trip) {
                Trip new_trip;
                new_trip.stops.push_back(del);
                fill_trip_data(best_h, new_trip,state);
                state.helis[best_h].trips.push_back(new_trip);
                state.helis[best_h].total_distance += new_trip.distance;
            } else {
                Trip& trip = state.helis[best_h].trips[best_t];
                trip.stops.insert(trip.stops.begin() + best_pos, del);
                trip.distance += best_extra_dist;
                trip.value += del_value;
                trip.weight += del_weight;
                state.helis[best_h].total_distance += best_extra_dist;
            }
            state.delivered[del.village - 1][0] += del.d;
            state.delivered[del.village - 1][1] += del.p;
            state.delivered[del.village - 1][2] += del.o;
        }
    }
    obj_func(state);
    return state;
}


void print_state(const State& s) {
    // Set floating point precision for cleaner output
    std::cout << std::fixed << std::setprecision(2);

    std::cout << "========================================\n";
    std::cout << "           STATE ANALYSIS\n";
    std::cout << "========================================\n\n";

    std::cout << ">> Overall Objective Score: " << s.objective << "\n\n";

    // --- Print details for each helicopter ---
    for (int h = 0; h < s.helis.size(); ++h) {
        const auto& helicopter = s.helis[h];
        const auto& heli_info = helis[h]; // Access global info

        std::cout << "--- Helicopter " << h << " (Home: City " << heli_info.home_city << ") ---\n";
        std::cout << "  Total Distance Flown: " << helicopter.total_distance << "\n";
        std::cout << "  Number of Trips: " << helicopter.trips.size() << "\n";

        for (int t = 0; t < helicopter.trips.size(); ++t) {
            const auto& trip = helicopter.trips[t];
             if (trip.stops.empty()) {
                // std::cout << "[EMPTY]\n";
                continue;
            }

            std::cout << "    - Trip " << t << ": ";
           
            // Print the full route
            std::cout << "Route: C" << heli_info.home_city;
            for (const auto& stop : trip.stops) {
                std::cout << " -> V" << stop.village;
            }
            std::cout << " -> C" << heli_info.home_city << "\n";

            // Print trip stats
            std::cout << "      Stats: Dist=" << trip.distance
                      << " | Value=" << trip.value
                      << " | Weight=" << trip.weight << "\n";
            
            // Print deliveries on this trip
            std::cout << "      Deliveries: ";
            for (const auto& stop : trip.stops) {
                 std::cout << "V" << stop.village << "->(d:" << stop.d 
                           << ",p:" << stop.p << ",o:" << stop.o << ") ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    // --- Print summary of total deliveries per village ---
    std::cout << "--- Total Delivered Packages Summary ---\n";
    bool any_delivered = false;
    for (int v = 0; v < s.delivered.size(); ++v) {
        if (s.delivered[v][0] > 0 || s.delivered[v][1] > 0 || s.delivered[v][2] > 0) {
            any_delivered = true;
            std::cout << "  Village " << v + 1 << ": (d:" << s.delivered[v][0]
                      << ", p:" << s.delivered[v][1]
                      << ", o:" << s.delivered[v][2] << ")\n";
        }
    }
    if (!any_delivered) {
        std::cout << "  No packages delivered in this state.\n";
    }

    std::cout << "========================================\n\n";
}
void calculate_full_objective(State &s) {
    s.objective = 0.0;
    double total_value = 0.0;
    double total_cost = 0.0;

    for (int h = 0; h < s.helis.size(); ++h) {
        for (const auto& trip : s.helis[h].trips) {
            if (!trip.stops.empty()) {
                total_cost += helis[h].F; // Add fixed cost for each non-empty trip
                total_cost += helis[h].alpha * trip.distance;
                total_value += trip.value;
            }
        }
    }
    s.objective = total_value - total_cost;
}
State generate_simple_coverage_state() {
    State state;
    state.helis.resize(H);
    state.delivered.assign(V, {0, 0, 0});

    // Loop through each village to try and serve it
    for (int v_idx = 0; v_idx < V; ++v_idx) {
        // Create a minimal delivery of just one "p" packet
        Delivery del = {v_idx + 1, 0, 1, 0};
        double del_weight = w_p; // Weight of one p_pack
        double del_value = v_p;   // Value of one p_pack

        // Find the first helicopter that can make a dedicated trip
        for (int h = 0; h < H; ++h) {
            double round_trip_dist = dist_matrix[helis[h].home_city - 1][C + v_idx] * 2;

            // Check all three feasibility constraints
            bool is_feasible = (
                round_trip_dist <= helis[h].dcap + 1e-9 &&
                del_weight <= helis[h].w_cap + 1e-9 &&
                state.helis[h].total_distance + round_trip_dist <= DMax + 1e-9
            );

            if (is_feasible) {
                // A helicopter was found! Create the new trip.
                Trip new_trip;
                new_trip.stops.push_back(del);
                new_trip.distance = round_trip_dist;
                new_trip.weight = del_weight;
                new_trip.value = del_value;

                // Update the state
                state.helis[h].trips.push_back(new_trip);
                state.helis[h].total_distance += round_trip_dist;
                state.delivered[v_idx][1] += 1; // Delivered one 'p' pack

                // This village is now covered, so we can break the inner loop
                // and move on to the next village.
                break;
            }
        }
    }

    // Calculate the final objective score for the generated state
    calculate_full_objective(state);
    return state;
}
// ===================================================================
// MAIN ALNS LOOP
// ===================================================================

State run_alns(State initial, ALNSData &alns){
    // obj_func(initial); // Calculate initial objective
    State current = initial;
    State best = initial;

    if(fabs(initial.objective) > 1e-9 && alns.T0 == 0){
        alns.T0 = std::max(1.0, 0.1 * fabs(initial.objective));
        alns.T = alns.T0;
    } else if (alns.T0 == 0) {
        alns.T0 = 100; // Fallback temperature
        alns.T = alns.T0;
    }


    int max_iter = 20000;
    for(int iter = 0; iter < max_iter; ++iter) {
        int d_idx = adaptive_picking(alns.weightD, alns.rng);
        int r_idx = adaptive_picking(alns.weightR, alns.rng);
        alns.usesD[d_idx]++;
        alns.usesR[r_idx]++;

        State temp = current;
        Pool pool;

        int total_stops = 0;
        for(const auto& h : temp.helis) for(const auto& t : h.trips) total_stops += t.stops.size();
        if (total_stops == 0) continue;
        
        std::uniform_int_distribution<int> remove_dist(std::max(1, total_stops / 10), std::max(2, total_stops / 3));
        int num_to_remove = remove_dist(alns.rng);
        // pool = destroy_shaw(temp, num_to_remove, alns.rng);
// pool = destroy_random_stop(temp, num_to_remove, alns.rng);
            // cout<<alns.destroy_names[d_idx];

        switch(alns.destroy_names[d_idx]) {
            case random_stop: pool = destroy_random_stop(temp, num_to_remove, alns.rng); break;
            case route_remove: pool = destroy_route_remove(temp, num_to_remove, alns.rng); break;
            case shaw: pool = destroy_shaw(temp, num_to_remove, alns.rng); break;
            case worst_values_destroyed: pool = destroy_worst_value(temp, num_to_remove, alns.rng); break;
            case perishable_punished: pool = destroy_perishable_aware(temp, num_to_remove, alns.rng); break;
        }
        //   repair_random_insert(temp, pool, alns.rng); 
        // repair_greedy_insert(temp, pool, alns.rng);
        //    cout<<alns.repair_names[r_idx]<<endl;

        switch(alns.repair_names[r_idx]) {
            case greedy_insert: repair_greedy_insert(temp, pool, alns.rng); break;
            // case regret2_insert: repair_regret2_insert(temp, pool, alns.rng); break;
            case cluster_build: repair_cluster_build(temp, pool, alns.rng); break;
            case random_insert: repair_random_insert(temp, pool, alns.rng); break;
            case new_insert: repair_and_adjust_quantities(temp,pool,alns.rng); break;
        }
        // print_state(temp);

        double delta = temp.objective - current.objective;
        bool accept = (delta >= 0) || (std::uniform_real_distribution<double>(0,1)(alns.rng) < exp(delta / alns.T));

        if (accept) {
            if (temp.objective > best.objective) {
                best = temp;
                alns.scoreD[d_idx] += alns.r_best;
                alns.scoreR[r_idx] += alns.r_best;
            } else if(temp.objective > current.objective){
                alns.scoreD[d_idx] += alns.r_good;
                alns.scoreR[r_idx] += alns.r_good;
            }
            else{
                   alns.scoreD[d_idx] += alns.r_accepted;
                alns.scoreR[r_idx] += alns.r_accepted;
            }
                            current = temp;
                 if (std::uniform_real_distribution<>(0, 1)(alns.rng) < 0.20) {
                change_quantity(current, alns.rng); 
                swap_packet_type(current,alns.rng);
                // We don't need to check the return value; if it makes a change, 
                // the 'current' state is now even better.
            }

        }


        if ((iter + 1) % alns.update_window == 0) {
            for (int i = 0; i < (int)alns.weightD.size(); i++) {
                if (alns.usesD[i] > 0) alns.weightD[i] = std::max(0.01, (1.0 - alns.rho) * alns.weightD[i] + alns.rho * alns.scoreD[i] / alns.usesD[i]);
                alns.scoreD[i] = 0.0; alns.usesD[i] = 0.0;
            }
            for (int i = 0; i < (int)alns.weightR.size(); i++) {
                if (alns.usesR[i] > 0) alns.weightR[i] = std::max(0.01, (1.0 - alns.rho) * alns.weightR[i] + alns.rho * alns.scoreR[i] / alns.usesR[i]);
                alns.scoreR[i] = 0.0; alns.usesR[i] = 0.0;
            }
        }
        alns.T *= 0.9995;
    }
    return best;
}


void read_input_from_file(const std::string& filename) {
    // Create an input file stream object to read from the specified file
    std::ifstream inputFile(filename);

    // Check if the file was opened successfully. If not, print an error and exit.
    if (!inputFile.is_open()) {
        std::cerr << "Error: Could not open the file '" << filename << "'." << std::endl;
        exit(1); // Exit the program with an error code
    }

    // Now, simply use 'inputFile' exactly as you were using 'std::cin'
    double time_limit;
    inputFile >> time_limit; // First line is time, but we don't use it here

    inputFile >> DMax;
    inputFile >> w_d >> v_d >> w_p >> v_p >> w_o >> v_o;

    inputFile >> C;
    cities.resize(C);
    for (int i = 0; i < C; ++i) {
        inputFile >> cities[i].first >> cities[i].second;
    }

    inputFile >> V;
    villages.resize(V);
    for (int i = 0; i < V; ++i) {
        inputFile >> villages[i].x >> villages[i].y >> villages[i].n;
    }

    inputFile >> H;
    helis.resize(H);
    for (int i = 0; i < H; ++i) {
        inputFile >> helis[i].home_city >> helis[i].w_cap >> helis[i].dcap >> helis[i].F >> helis[i].alpha;
    }

    // The file is automatically closed when 'inputFile' goes out of scope.
    std::cout << "Successfully read all data from '" << filename << "'." << std::endl;
}
// ===================================================================
// MAIN FUNCTION
// ===================================================================

int main(){
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);
    read_input_from_file("input.txt");
    // double time;
    // std::cin >> time;

    // std::cin >> DMax;
    // std::cin >> w_d >> v_d >> w_p >> v_p >> w_o >> v_o;

    // std::cin >> C;
    // cities.resize(C);
    // for (int i = 0; i < C; ++i) {
    //     std::cin >> cities[i].first >> cities[i].second;
    // }

    // std::cin >> V;
    // villages.resize(V);
    // for (int i = 0; i < V; ++i) {
    //     std::cin >> villages[i].x >> villages[i].y >> villages[i].n;
    // }

    // std::cin >> H;
    // helis.resize(H);
    // for (int i = 0; i < H; ++i) {
    //     std::cin >> helis[i].home_city >> helis[i].w_cap >> helis[i].dcap >> helis[i].F >> helis[i].alpha;
    // }
    
    precompute_distances();
    
    State initial_state = generate_initial_state();
    print_state(initial_state);
    
    // pool = destroy_random_stop(initial_state, num_to_remove, alns.rng);

    ALNSData alns;
    alns.rng.seed(0); 

    alns.destroy_names = { random_stop, route_remove, shaw, worst_values_destroyed, perishable_punished };
    alns.repair_names = { greedy_insert, cluster_build, random_insert, new_insert };

    alns.weightD.assign(alns.destroy_names.size(), 1.0);
    alns.scoreD.assign(alns.destroy_names.size(), 0.0);
    alns.usesD.assign(alns.destroy_names.size(), 0.0);

    alns.weightR.assign(alns.repair_names.size(), 1.0);
    alns.scoreR.assign(alns.repair_names.size(), 0.0);
    alns.usesR.assign(alns.repair_names.size(), 0.0);

    State best_solution = run_alns(initial_state, alns);
    print_state(best_solution);
    cout<<(best_solution.objective);
    // double result = obj_func(best_solution);
    // std::cout << "Objective value: " << result << std::endl;
    // // ... Code to print the final solution in the required format ...

    return 0;
}
