#ifndef TAXI_H
#define TAXI_H

#include <stdbool.h>
#include <stdlib.h>

// R, G, B, Y Locations
static const int LOCS[4][2] = {{0, 0}, {0, 4}, {4, 0}, {4, 3}};

typedef struct {
	float *observations;
	int *actions;
	float *rewards;
	unsigned char *terminals;

	// Game State
	int taxi_row;
	int taxi_col;
	int pass_idx; // 0-3 (Locs), 4 (In Taxi)
	int dest_idx; // 0-3 (Locs)

	int max_steps;
	int current_step;
	int num_agents;

	void *client;
} CTaxiEnv;

void allocate_taxi(CTaxiEnv *env);
void free_taxi(CTaxiEnv *env);
void reset_taxi(CTaxiEnv *env);
void step_taxi(CTaxiEnv *env);
void render_taxi(CTaxiEnv *env);

// Helper to convert game state to integer (0-499)
int encode_state(int row, int col, int pass, int dest);

#endif
