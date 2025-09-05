#include <iostream>
#include <vector>
#include <cmath>
#include <array>
#include <algorithm>
#include <random>
#include <numeric>
#include <iomanip>
#include <fstream>
#include<bits/stdc++.h>
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
    new_insert,
    repair_demand,
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
        
    }
    return trip_value;
}

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
// This helper calculates the TRUE value a new delivery will add,
// respecting the per-village food limit.
double calculate_effective_value_change(const State& s, const Delivery& del) {
    int v_idx = del.village - 1;
    
    // Get the village's food limit
    int food_limit = 9 * villages[v_idx].n;

    // Find out how much food has already been delivered
    int current_food = s.delivered[v_idx][0] + s.delivered[v_idx][1];

    // Calculate how much more food the village can take before hitting the limit
    int food_room_left = std::max(0, food_limit - current_food);

    // This delivery adds a certain amount of new food
    int food_in_delivery = del.d + del.p;

    // The "useful" new food is the smaller of what's in the delivery vs. what's needed
    int useful_added_food = std::min(food_in_delivery, food_room_left);

    double effective_value = 0.0;

    // Calculate the value of the useful added food, prioritizing perishable ('p')
    if (useful_added_food > 0) {
        int useful_p = std::min(del.p, useful_added_food);
        effective_value += useful_p * v_p;
        
        int remaining_useful_food = useful_added_food - useful_p;
        effective_value += remaining_useful_food * v_d;
    }

    // Add the value of 'other' packets (assuming they have no limit)
    effective_value += del.o * v_o;

    return effective_value;
}
// Helper function to calculate the total effective value from a village's deliveries
// This is needed by the function below.
double calculate_value_for_village(const std::array<int, 3>& delivered_packets, int v_idx) {
    int food_limit = 9 * villages[v_idx].n;
    int total_food = delivered_packets[0] + delivered_packets[1];
    int useful_food = std::min(total_food, food_limit);
    
    double value = 0.0;
    int perishable = delivered_packets[1];
    if (useful_food <= perishable) {
        value += useful_food * v_p;
    } else {
        value += perishable * v_p;
        value += (useful_food - perishable) * v_d;
    }
    value += delivered_packets[2] * v_o;
    return value;
}

// This calculates the TRUE change in value when a delivery is REMOVED.
double calculate_effective_value_change_on_removal(const State& s, const Delivery& del) {
    int v_idx = del.village - 1;
    
    // 1. Calculate the village's total value BEFORE the removal
    double value_before = calculate_value_for_village(s.delivered[v_idx], v_idx);

    // 2. Simulate the removal
    std::array<int, 3> delivered_after = s.delivered[v_idx];
    delivered_after[0] -= del.d;
    delivered_after[1] -= del.p;
    delivered_after[2] -= del.o;

    // 3. Calculate the village's total value AFTER the removal
    double value_after = calculate_value_for_village(delivered_after, v_idx);

    // 4. The change is the difference
    return value_after - value_before; // This will be negative or zero
}

// ===================================================================
// DESTROY OPERATORS
// ===================================================================
void remove_stop_from_trip(State& s, Pool& pool, const Stop_tracking& ref) {
    Trip &trip = s.helis[ref.h].trips[ref.t];
    Delivery del = trip.stops[ref.s];

    pool.removed.push_back({del.village, del.d, del.p, del.o});
    
    // --- FIX: Calculate the true change in value BEFORE updating state ---
    double effective_value_change = calculate_effective_value_change_on_removal(s, del);
    
    // Update delivered counts AFTER calculating the change
    s.delivered[del.village - 1][0] -= del.d;
    s.delivered[del.village - 1][1] -= del.p;
    s.delivered[del.village - 1][2] -= del.o;
    
    int prev_node = (ref.s == 0) ? helis[ref.h].home_city - 1 : C + trip.stops[ref.s - 1].village - 1;
    int current_node = C + del.village - 1;
    int next_node = (ref.s == (int)trip.stops.size() - 1) ? helis[ref.h].home_city - 1 : C + trip.stops[ref.s + 1].village - 1;

    double dist_change = dist_matrix[prev_node][next_node] - (dist_matrix[prev_node][current_node] + dist_matrix[current_node][next_node]);
    double cost_change = helis[ref.h].alpha * dist_change;
    
    // Use the corrected effective_value_change for the objective update.
    // The change is (value_change - cost_change).
    s.objective += (effective_value_change - cost_change);

    // Update other cached values
    double raw_del_value = del.d * v_d + del.p * v_p + del.o * v_o;
    trip.distance += dist_change;
    s.helis[ref.h].total_distance += dist_change;
    trip.value -= raw_del_value; // Trip cache uses raw value
    trip.weight -= (del.d * w_d + del.p * w_p + del.o * w_o);

    trip.stops.erase(trip.stops.begin() + ref.s);
    if (trip.stops.empty()) {
        s.objective += helis[ref.h].F;
    }
}
#include <climits>
Delivery allocate_best_delivery(int village_id, State &s, int heli_id,
                                     double spare_weight,
                                     int max_d = INT_MAX, int max_p = INT_MAX, int max_o = INT_MAX) {
    int v_idx = village_id - 1;
    int people = villages[v_idx].n;

    int food_limit = 9 * people;
    int other_limit = people;

    int already_d = s.delivered[v_idx][0];
    int already_p = s.delivered[v_idx][1];
    int already_o = s.delivered[v_idx][2];

    int food_remaining = std::max(0, food_limit - (already_d + already_p));
    int other_remaining = std::max(0, other_limit - already_o);

    Delivery best = {village_id, 0, 0, 0};
    double best_value = -1;

    if (spare_weight + 1e-12 < std::min({w_d, w_p, w_o})) return best;
    if (food_remaining <= 0 && other_remaining <= 0) return best;

    std::vector<std::vector<int>> perms = {
        {1,0,2}
    };

    for (auto &order : perms) {
        double sw = spare_weight;
        int fd = std::min(food_remaining, max_d);
        int fp = std::min(food_remaining, max_p);
        int fo = std::min(other_remaining, max_o);

        Delivery cur = {village_id, 0, 0, 0};

        for (int t : order) {
            if (t == 0) { // dry
                int give = std::min(fd, (int)(sw / w_d));
                cur.d += give;
                sw -= give * w_d;
                fd -= give;
                food_remaining -= give;
                fp = std::min(fp, food_remaining);
            } else if (t == 1) { // perishable
                int give = std::min(fp, (int)(sw / w_p));
                cur.p += give;
                sw -= give * w_p;
                fp -= give;
                food_remaining -= give;
                fd = std::min(fd, food_remaining);
            } else { // other
                int give = std::min(fo, (int)(sw / w_o));
                cur.o += give;
                sw -= give * w_o;
                fo -= give;
            }
        }

        double val = cur.d * v_d + cur.p * v_p + cur.o * v_o;
        if (val > best_value + 1e-12) {
            best_value = val;
            best = cur;
        }
    }

    return best;
}

// Greedy allocator that picks packet types by value-per-weight density,
// respects village limits, current delivered counts, helicopter spare weight,
// and (optionally) per-type availability limits (max_d/max_p/max_o).
Delivery allocate_best_delivery(int village_id, State &s, int heli_id,
                                double spare_weight,
                                const std::vector<int> &order,
                                int max_d = INT_MAX, int max_p = INT_MAX, int max_o = INT_MAX) {
    int v_idx = village_id - 1;
    int people = villages[v_idx].n;

    int food_limit = 9 * people;
    int other_limit = people;

    int already_d = s.delivered[v_idx][0];
    int already_p = s.delivered[v_idx][1];
    int already_o = s.delivered[v_idx][2];

    int food_remaining = std::max(0, food_limit - (already_d + already_p));
    int other_remaining = std::max(0, other_limit - already_o);

    Delivery best = {village_id, 0, 0, 0};
    double best_value = -1;

    if (spare_weight + 1e-12 < std::min({w_d, w_p, w_o})) return best;
    if (food_remaining <= 0 && other_remaining <= 0) return best;

    double sw = spare_weight;
    int fd = std::min(food_remaining, max_d);
    int fp = std::min(food_remaining, max_p);
    int fo = std::min(other_remaining, max_o);

    Delivery cur = {village_id, 0, 0, 0};

    for (int t : order) {
        if (t == 0) { // dry
            int give = std::min(fd, (int)(sw / w_d));
            cur.d += give; sw -= give * w_d;
            fd -= give; food_remaining -= give;
            fp = std::min(fp, food_remaining);
        } else if (t == 1) { // perishable
            int give = std::min(fp, (int)(sw / w_p));
            cur.p += give; sw -= give * w_p;
            fp -= give; food_remaining -= give;
            fd = std::min(fd, food_remaining);
        } else { // other
            int give = std::min(fo, (int)(sw / w_o));
            cur.o += give; sw -= give * w_o;
            fo -= give;
        }
    }

    double val = cur.d * v_d + cur.p * v_p + cur.o * v_o;
    if (val > best_value + 1e-12) {
        best_value = val;
        best = cur;
    }

    return best;
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
    for (auto& h : s.helis) { h.trips.erase(remove_if(h.trips.begin(), h.trips.end(), [](const Trip& t){ return t.stops.empty(); }), h.trips.end()); }
    return pool;
}

Pool destroy_route_remove(State & s, int num_remove, std::mt19937 & rng) {
    Pool pool;
    std::vector<std::pair<int,int>> candidates;
    for (int h = 0; h < (int)s.helis.size(); h++) for (int t = 0; t < (int)s.helis[h].trips.size(); t++) if (!s.helis[h].trips[t].stops.empty()) candidates.push_back({h, t});
    if (candidates.empty()) return pool;

    shuffle(candidates.begin(), candidates.end(), rng);
    int remove_count = std::min(num_remove, (int)candidates.size());
// --- FIX PART 1: Collect trips to remove instead of modifying immediately ---
    std::vector<std::pair<int, int>> trips_to_remove;
    for (int k = 0; k < remove_count; k++) {
        trips_to_remove.push_back(candidates[k]);
    }

    // Sort the indices in descending order to allow for safe removal
    sort(trips_to_remove.begin(), trips_to_remove.end(), [](const auto& a, const auto& b) {
        if (a.first != b.first) return a.first > b.first; // Helicopter index
        return a.second > b.second; // Trip index
    });

    // --- FIX PART 2: Now, iterate through the sorted list and safely remove ---
    for (const auto& ref : trips_to_remove) {
        int h = ref.first;
        int t = ref.second;
        Trip& trip = s.helis[h].trips[t];

        // Perform all state updates as before
        // double effective_trip_value = calculate_effective_trip_value(s, trip);
        // s.objective -= (effective_trip_value - (helis[h].alpha * trip.distance + helis[h].F));
        
        for (const Delivery &del : trip.stops) {
            pool.removed.push_back({del.village, del.d, del.p, del.o});
            // s.delivered[del.village - 1][0] -= del.d;
            // s.delivered[del.village - 1][1] -= del.p;
            // s.delivered[del.village - 1][2] -= del.o;
        }
        
        // s.helis[h].total_distance -= trip.distance;

        // Erase the trip from the vector
        s.helis[h].trips.erase(s.helis[h].trips.begin() + t);
    }
    for (auto& h : s.helis) { h.trips.erase(remove_if(h.trips.begin(), h.trips.end(), [](const Trip& t){ return t.stops.empty(); }), h.trips.end()); }

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
    for (auto& h : s.helis) { h.trips.erase(remove_if(h.trips.begin(), h.trips.end(), [](const Trip& t){ return t.stops.empty(); }), h.trips.end()); }
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
    for (auto& h : s.helis) { h.trips.erase(remove_if(h.trips.begin(), h.trips.end(), [](const Trip& t){ return t.stops.empty(); }), h.trips.end()); }
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
    for (auto& h : s.helis) { h.trips.erase(remove_if(h.trips.begin(), h.trips.end(), [](const Trip& t){ return t.stops.empty(); }), h.trips.end()); }
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
    // if (p_to_add > 0 || o_to_add > 0) {
    //     obj_func(s); // Use the full objective function to re-sync the score
    //     return true;
    // }

    return false;
}
void random_village_fill(State &s, std::mt19937 &rng,vector<int> order) {
    // 1. Collect villages with remaining demand
    std::vector<int> candidates;
    for (int v = 0; v < V; v++) {
        int people = villages[v].n;
        int food_limit = 9 * people;
        int other_limit = 1 * people;

        int already_d = s.delivered[v][0];
        int already_p = s.delivered[v][1];
        int already_o = s.delivered[v][2];

        if (already_d + already_p < food_limit || already_o < other_limit) {
            candidates.push_back(v + 1); // 1-based village id
        }
    }
    if (candidates.empty()) return;

    // 2. Shuffle and decide how many villages to pick
    shuffle(candidates.begin(), candidates.end(), rng);
    std::uniform_int_distribution<int> how_many(1, (int)candidates.size());
    int pick_count = how_many(rng);

    for (int i = 0; i < pick_count; i++) {
        int village_id = candidates[i];

        double best_gain = -1e18;
        int best_h = -1;
        Delivery best_del;
        double best_trip_dist = 0.0;

        // 3. Evaluate all helicopters greedily
        for (int h = 0; h < H; h++) {
            double spare_weight = helis[h].w_cap;
            Delivery del = allocate_best_delivery(village_id, s, h, spare_weight,order);

            if (del.d == 0 && del.p == 0 && del.o == 0) continue;

            // round trip distance
            double trip_dist = dist_matrix[helis[h].home_city - 1][C + village_id - 1] * 2;

            // feasibility checks
            if (trip_dist > helis[h].dcap + 1e-9) continue;
            if (s.helis[h].total_distance + trip_dist > DMax + 1e-9) continue;

            // compute value and cost
            double raw_val = del.d * v_d + del.p * v_p + del.o * v_o;
            double gain = raw_val - (helis[h].alpha * trip_dist + helis[h].F);

            if (gain > best_gain) {
                best_gain = gain;
                best_h = h;
                best_del = del;
                best_trip_dist = trip_dist;
            }
        }

        if (best_h == -1) continue; // no feasible helicopter found

        // 4. Build the trip with the best helicopter
        Trip new_trip;
        new_trip.stops.push_back(best_del);
        fill_trip_data(best_h, new_trip, s);

        s.helis[best_h].trips.push_back(new_trip);
        s.helis[best_h].total_distance += best_trip_dist;

        s.delivered[village_id - 1][0] += best_del.d;
        s.delivered[village_id - 1][1] += best_del.p;
        s.delivered[village_id - 1][2] += best_del.o;

        s.objective += best_gain;
    }
}

// This is a local search operator that tries to swap packet types within a delivery.
// Returns 'true' if it successfully made a change.

struct BestTriple {
    int d;
    int p;
    int o;
    double value; // raw value = d*v_d + p*v_p + o*v_o
};

// Compute best (d,p,o) under:
// - food_rem = how many more food packets (d+p) we may send to this village (>=0).
// - other_rem = how many more other packets (o) we may send (>=0).
// - wcap = helicopter spare weight available.
// It searches all feasible (d,p,o) with d+p <= food_rem and o <= other_rem and weight <= wcap
BestTriple optimal_allocation(int food_rem, int other_rem, double wcap) {
    BestTriple best{0,0,0, -1e18};

    // food_rem may be large; but usual problems have moderate n. Complexity O(food_rem * other_rem).
    for (int total_food = 0; total_food <= food_rem; ++total_food) {
        // We will split total_food into d and p.
        for (int o = 0; o <= other_rem; ++o) {
            // We need to choose d and p such that d+p = total_food.
            // But iterate p from 0..total_food (d = total_food - p)
            for (int p = 0; p <= total_food; ++p) {
                int d = total_food - p;
                double weight = d * w_d + p * w_p + o * w_o;
                if (weight > wcap + 1e-12) continue;
                double val = d * v_d + p * v_p + o * v_o;
                if (val > best.value) {
                    best = {d, p, o, val};
                }
            }
        }
    }

    // if nothing fits, return zero triple
    if (best.value < -1e17) return BestTriple{0,0,0,0.0};
    return best;
}

void swap_packet_type(State &s, std::mt19937 &rng) { 
    std::vector<int> candidates;
    for (int v = 0; v < V; v++) {
        int people = villages[v].n;
        int food_limit = 9 * people;
        int other_limit = people;

        int already_d = s.delivered[v][0];
        int already_p = s.delivered[v][1];
        int already_o = s.delivered[v][2];

        if (already_d + already_p < food_limit || already_o < other_limit) {
            candidates.push_back(v);
        }
    }
    if (candidates.empty()) return;

    shuffle(candidates.begin(), candidates.end(), rng);
    std::uniform_int_distribution<int> how_many(1, (int)candidates.size());
    int pick_count = how_many(rng);

    for (int i = 0; i < pick_count; i++) {
        int v_idx = candidates[i];
        int people = villages[v_idx].n;

        int food_limit = 9 * people;
        int other_limit = people;

        int rem_d = food_limit - (s.delivered[v_idx][0] + s.delivered[v_idx][1]);
        int rem_o = other_limit - s.delivered[v_idx][2];

        // pick best helicopter
        int best_h = -1;
        double best_metric = 1e18;
        for (int h = 0; h < H; h++) {

           int city_id = helis[h].home_city;    // assuming each heli has a city index
double dist = dist_matrix[city_id][C + v_idx];
            double metric = dist + s.helis[h].total_distance / helis[h].dcap;
            if (metric < best_metric) {
                best_metric = metric;
                best_h = h;
            }
        }
        if (best_h == -1) continue;

        double cap_weight = helis[best_h].w_cap;

        // DP allocation
//        int food_limit = 9 * people;
// int other_limit = people;

int food_rem = std::max(0, food_limit - (s.delivered[v_idx][0] + s.delivered[v_idx][1]));
int other_rem = std::max(0, other_limit - s.delivered[v_idx][2]);

BestTriple alloc = optimal_allocation(food_rem, other_rem, cap_weight);
if (alloc.value <= 0) continue;

        // Build trip
        Trip new_trip;
        new_trip.stops.push_back({v_idx+1, alloc.d, alloc.p, alloc.o});
        fill_trip_data(best_h, new_trip, s);

        s.helis[best_h].trips.push_back(new_trip);
        s.helis[best_h].total_distance += new_trip.distance;
        s.delivered[v_idx][0] += alloc.d;
        s.delivered[v_idx][1] += alloc.p;
        s.delivered[v_idx][2] += alloc.o;
    }
}

// ===================================================================
// REPAIR OPERATORS
// ===================================================================


// This is a "demand-centric" repair operator. It aggregates all removed items
// into a total demand per village. Then, it iteratively finds the single most
// profitable NEW trip it can create for any helicopter to any village with
// remaining demand, builds it, and repeats until no more profitable trips can be made.
// CORRECTED VERSION
void repair_fill_demand_new_trips(State &s, Pool &pool, std::mt19937 &rng,vector<int> order) {
    if (pool.removed.empty()) return;

    // 1. Aggregate all removed deliveries into a total demand per village.
    std::map<int, std::array<int, 3>> remaining_demand;
    for (const auto& rem : pool.removed) {
        remaining_demand[rem.v][0] += rem.d;
        remaining_demand[rem.v][1] += rem.p;
        remaining_demand[rem.v][2] += rem.o;
    }

    // 2. Iteratively create the best possible new trip until no more profitable trips can be found.
    while (true) {
        double best_overall_gain = 1e-9; 
        int best_h = -1;
        int best_village_id = -1;
        Delivery best_delivery_for_trip;

        for (auto const& [village_id, demand] : remaining_demand) {
            if (demand[0] + demand[1] + demand[2] == 0) continue;

            for (int h = 0; h < (int)s.helis.size(); ++h) {
                double spare_weight = helis[h].w_cap;
                
                // --- FIX PART 1: ---
                // Call the allocator WITHOUT limiting it to the pool's demand.
                // Let it determine the TRUE need based on village limits and s.delivered.
                Delivery potential_del = allocate_best_delivery(village_id, s, h, spare_weight,order);

                if (potential_del.d == 0 && potential_del.p == 0 && potential_del.o == 0) {
                    continue;
                }
                
                // --- FIX PART 2: ---
                // NOW, cap the delivery to not exceed what's left in our demand pool for this repair step.
                // This prevents the operator from "creating" deliveries out of thin air.
                potential_del.d = std::min(potential_del.d, demand[0]);
                potential_del.p = std::min(potential_del.p, demand[1]);
                potential_del.o = std::min(potential_del.o, demand[2]);

                if (potential_del.d == 0 && potential_del.p == 0 && potential_del.o == 0) {
                    continue;
                }

                double trip_dist = dist_matrix[helis[h].home_city - 1][C + village_id - 1] * 2;
                if (trip_dist > helis[h].dcap + 1e-9 || s.helis[h].total_distance + trip_dist > DMax + 1e-9) {
                    continue;
                }

                double effective_value = calculate_effective_value_change(s, potential_del);
                double gain = effective_value - (helis[h].alpha * trip_dist) - helis[h].F;

                if (gain > best_overall_gain) {
                    best_overall_gain = gain;
                    best_h = h;
                    best_village_id = village_id;
                    best_delivery_for_trip = potential_del;
                }
            }
        }

        if (best_h == -1) {
            break; 
        }

        // --- The rest of the function is the same ---
        Trip new_trip;
        new_trip.stops.push_back(best_delivery_for_trip);
        fill_trip_data(best_h, new_trip, s); 
        
        s.helis[best_h].trips.push_back(new_trip);

        s.objective += best_overall_gain;
        s.helis[best_h].total_distance += new_trip.distance;
        s.delivered[best_village_id - 1][0] += best_delivery_for_trip.d;
        s.delivered[best_village_id - 1][1] += best_delivery_for_trip.p;
        s.delivered[best_village_id - 1][2] += best_delivery_for_trip.o;

        remaining_demand[best_village_id][0] -= best_delivery_for_trip.d;
        remaining_demand[best_village_id][1] -= best_delivery_for_trip.p;
        remaining_demand[best_village_id][2] -= best_delivery_for_trip.o;
    }
    
    pool.removed.clear();
}
void repair_greedy_insert(State &s, Pool &pool, std::mt19937 &rng) {
    shuffle(pool.removed.begin(), pool.removed.end(), rng);
    auto it = pool.removed.begin();
    while (it != pool.removed.end()) {
        Delivery del{it->v, it->d, it->p, it->o};
        // We still need del_weight for constraint checking
        double del_weight = del.d * w_d + del.p * w_p + del.o * w_o;

        double best_gain = -1e18;
        int best_h = -1, best_t = -1, best_pos = -1;
        double best_extra_dist = 0.0;

        // --- FIX PART 1: Calculate true effective value for gain calculation ---
        double effective_value = calculate_effective_value_change(s, del);

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

                    // Use the corrected 'effective_value' here
                    double gain = effective_value - helis[h].alpha * extra_dist;
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

            // Note: trip.value should be updated with the RAW value of packets,
            // as it's just a cache for that trip. The global objective uses the true gain.
            double raw_del_value = del.d * v_d + del.p * v_p + del.o * v_o;
            trip.distance += best_extra_dist;
            trip.value += raw_del_value; 
            trip.weight += del_weight;

            s.helis[best_h].total_distance += best_extra_dist;
            s.delivered[del.village - 1][0] += del.d; s.delivered[del.village - 1][1] += del.p; s.delivered[del.village - 1][2] += del.o;
            s.objective += best_gain; // This now uses the CORRECT gain
            inserted = true;
        } else {
            double best_new_trip_gain = -1e18; int best_h_for_new_trip = -1;
            for (int h = 0; h < (int)s.helis.size(); h++) {
                double trip_dist = dist_matrix[helis[h].home_city-1][C+del.village-1] * 2;
                if (trip_dist <= helis[h].dcap + 1e-9 && del_weight <= helis[h].w_cap + 1e-9 && s.helis[h].total_distance + trip_dist <= DMax + 1e-9) {
                    
                    // --- FIX PART 2: Use effective_value for new trip gain ---
                    double new_trip_gain = effective_value - helis[h].alpha * trip_dist - helis[h].F;
                    if (new_trip_gain > best_new_trip_gain) {
                        best_new_trip_gain = new_trip_gain; best_h_for_new_trip = h;
                    }
                }
            }
            if (best_h_for_new_trip != -1) {
                int h = best_h_for_new_trip;
                Trip new_trip; new_trip.stops.push_back(del);
                
                double raw_del_value = del.d * v_d + del.p * v_p + del.o * v_o;
                new_trip.weight = del_weight;
                new_trip.value = raw_del_value;
                new_trip.distance = dist_matrix[helis[h].home_city-1][C+del.village-1] * 2;
                
                s.helis[h].trips.push_back(new_trip);
                s.helis[h].total_distance += new_trip.distance;
                s.delivered[del.village-1][0] += del.d; s.delivered[del.village-1][1] += del.p; s.delivered[del.village-1][2] += del.o;
                s.objective += best_new_trip_gain; // This now uses the CORRECT gain
                inserted = true;
            }
        }

        if (inserted) {
            it = pool.removed.erase(it);
        } else {
            ++it;
        }
    }
    for (auto& h : s.helis) { h.trips.erase(remove_if(h.trips.begin(), h.trips.end(), [](const Trip& t){ return t.stops.empty(); }), h.trips.end()); }
}
void repair_random_insert(State &s, Pool &pool, std::mt19937 &rng) {
    shuffle(pool.removed.begin(), pool.removed.end(), rng);
    auto it = pool.removed.begin();
    while(it != pool.removed.end()){
        Delivery del{it->v, it->d, it->p, it->o};
        // We still need the raw value and weight for trip caches and constraints
        double raw_del_value = del.d*v_d + del.p*v_p + del.o*v_o;
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
            
            // --- FIX: Calculate the true effective value BEFORE updating the objective ---
            double effective_value = calculate_effective_value_change(s, del);

            // Now perform the insertion and update the state
            Trip& trip = s.helis[spot.h].trips[spot.t];
            trip.stops.insert(trip.stops.begin()+spot.pos, del);
            trip.distance += spot.extra_dist;
            trip.value += raw_del_value; // Trip cache uses raw value
            trip.weight += del_weight;
            
            s.helis[spot.h].total_distance += spot.extra_dist;
            s.delivered[del.village-1][0]+=del.d; s.delivered[del.village-1][1]+=del.p; s.delivered[del.village-1][2]+=del.o;
            
            // Use the corrected effective_value for the objective update
            s.objective += effective_value - helis[spot.h].alpha * spot.extra_dist;
            inserted = true;
        }

        if (inserted) {
            it = pool.removed.erase(it);
        } else {
            ++it;
        }
    }
    for (auto& h : s.helis) { h.trips.erase(remove_if(h.trips.begin(), h.trips.end(), [](const Trip& t){ return t.stops.empty(); }), h.trips.end()); }
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
    for (auto& h : s.helis) { h.trips.erase(remove_if(h.trips.begin(), h.trips.end(), [](const Trip& t){ return t.stops.empty(); }), h.trips.end()); }
}


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
    for (auto& h : s.helis) { h.trips.erase(remove_if(h.trips.begin(), h.trips.end(), [](const Trip& t){ return t.stops.empty(); }), h.trips.end()); }
}

// ===================================================================
// FULL OBJECTIVE FUNCTION (FOR VALIDATION & INITIALIZATION)
// ===================================================================

// State generate_randomized_initial_state() {
//     State state;
//     state.helis.resize(H);

//     vector<int> order(V);
//     iota(order.begin(), order.end(), 0);

//     std::random_device rd;
//     std::mt19937 g(rd());
//     std::shuffle(order.begin(), order.end(), g);

//     for (int h=0; h<H; h++) {
//         vector<int> assigned;
//         for (int v: order) if (rand()%H==h) assigned.push_back(v);
//         if (assigned.empty()) continue;

//         std::shuffle(assigned.begin(), assigned.end(), g);

//         vector<vector<int>> trips;
//         vector<int> cur;
//         double cap=0;
//         for (int v: assigned) {
//             if (cap > helis[h].w_cap*0.8 && !cur.empty()) {
//                 trips.push_back(cur); cur.clear(); cap=0;
//             }
//             cur.push_back(v);
//             cap += villages[v].n*5;
//         }
//         if (!cur.empty()) trips.push_back(cur);

//         for (auto& tv: trips) {
//             two_opt_people_weighted(tv, helis[h].home_city);

//             Trip trip;
//             double cap_left = helis[h].w_cap;
//             for (int v: tv) {
//                 Delivery del = allocate_packages(v, cap_left);
//                 if (del.d+del.p+del.o > 0) {
//                     trip.stops.push_back(del);
//                     cap_left -= del.d*w_d + del.p*w_p + del.o*w_o;
//                 }
//                 if (cap_left <= 0) break;
//             }
//             fill_trip_data(h,trip);
//             state.helis[h].trips.push_back(trip);
//         }
//         fill_helicopter_data(state.helis[h]);
//     }
//     fill_delivered_totals(state);
//     obj_func(state);
//     return state;
// }
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
    for (const auto& heli : state.helis) {
        for (const auto& trip : heli.trips) {
            for (const auto& del : trip.stops) {
                state.delivered[del.village - 1][0] += del.d;
                state.delivered[del.village - 1][1] += del.p;
                state.delivered[del.village - 1][2] += del.o;
            }
        }
    }
    
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

State generate_initial_state(const std::vector<int> &order) {
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
        
      // Compute how much spare weight this helicopter has
double spare_weight = helis[0].w_cap;  

// Use the new allocator to decide d, p, o
Delivery del = allocate_best_delivery(v_idx + 1, state, 0, spare_weight,order);

// Skip if nothing can be delivered
if (del.d + del.p + del.o == 0) continue;

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
           
            std::cout << "    - Trip " << t << ": ";
              if (trip.stops.empty()) {
                std::cout << "[EMPTY]\n";
                continue;
            }

           
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

// ===================================================================
// MAIN ALNS LOOP
// ===================================================================

State run_alns(State initial, ALNSData &alns,vector<int> order){
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


    int max_iter = 200000;
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
            case repair_demand: repair_fill_demand_new_trips(temp,pool,alns.rng,order); break;
        }
        // print_state(temp);
                change_quantity(current, alns.rng); 

                random_village_fill(temp,alns.rng,order);
                
        
                double true_score = obj_func(temp); 
                if(true_score == -1e18){continue;}

        //    if (std::abs(incremental_score - true_score) > 0.01) { // Use a small tolerance
        //     std::cout << "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
        //     std::cout << ">>> INCONSISTENCY DETECTED in iteration " << iter << "\n";
        //     std::cout << "    Used Destroy: " << alns.destroy_names[d_idx] 
        //               << " | Used Repair: " << alns.repair_names[r_idx] << "\n";
        //     std::cout << "--------------------------------------------------------\n";
        //     std::cout << "  - Score after incremental updates: " << incremental_score << "\n";
        //     std::cout << "  - True score from obj_func():      " << true_score << "\n";
        //     std::cout << "  - DIFFERENCE:                      " << (incremental_score - true_score) << "\n";
        //     std::cout << "\n>>> THIS MEANS A DESTROY OR REPAIR FUNCTION HAS A BUG <<<\n";
        //     std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n";
        //     // Optional: Pause the program here to inspect the state
        // }
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
            //      if (std::uniform_real_distribution<>(0, 1)(alns.rng) < 0.90) {
            //     // swap_packet_type(current,alns.rng);
            //     // We don't need to check the return value; if it makes a change, 
            //     // the 'current' state is now even better.
            // }

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

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);
    read_input_from_file("input8.txt");
    precompute_distances();

    ALNSData alns;
    alns.rng.seed(0);
    alns.destroy_names = { random_stop, route_remove, shaw, worst_values_destroyed, perishable_punished };
    alns.repair_names = { greedy_insert, cluster_build, random_insert, new_insert, repair_demand };

    alns.weightD.assign(alns.destroy_names.size(), 1.0);
    alns.scoreD.assign(alns.destroy_names.size(), 0.0);
    alns.usesD.assign(alns.destroy_names.size(), 0.0);

    alns.weightR.assign(alns.repair_names.size(), 1.0);
    alns.scoreR.assign(alns.repair_names.size(), 0.0);
    alns.usesR.assign(alns.repair_names.size(), 0.0);

    std::vector<std::vector<int>> perms = {
        {0,1,2}, {0,2,1},
        {1,0,2}, {1,2,0},
        {2,0,1}, {2,1,0}
    };

State best_overall = run_alns(generate_initial_state(perms[0]),alns, perms[0]);
print_state(best_overall);   
 for (size_t i = 1; i < perms.size(); i++) {
    State init = generate_initial_state(perms[i]);
    State sol = run_alns(init, alns,perms[i]);
    // cout << sol.objective << "\n";
    print_state(sol);
    if (sol.objective > best_overall.objective) {
        best_overall = sol;
    }
}

    print_state(best_overall);
    std::cout << best_overall.objective;
    return 0;
}
