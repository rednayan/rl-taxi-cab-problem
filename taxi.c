#include "taxi.h"
#include <stdlib.h>
#include <string.h>

static const int V_WALLS[5][5] = {
    {0, 1, 0, 0, 0}, // Row 0
    {0, 1, 0, 0, 0}, // Row 1
    {0, 0, 0, 0, 0}, // Row 2
    {0, 1, 0, 1, 0}, // Row 3
    {0, 1, 0, 1, 0}  // Row 4
};

int encode_state(int row, int col, int pass, int dest)
{
	// ((row * 5 + col) * 5 + pass) * 4 + dest
	int i = row;
	i = i * 5 + col;
	i = i * 5 + pass;
	i = i * 4 + dest;
	return i;
}

void compute_obs(CTaxiEnv *env)
{
	for (int i = 0; i < env->num_agents; i++) {
		int offset = i * 4;
		env->observations[offset + 0] = (float)env->taxi_row;
		env->observations[offset + 1] = (float)env->taxi_col;
		env->observations[offset + 2] = (float)env->pass_idx;
		env->observations[offset + 3] = (float)env->dest_idx;
	}
}

void allocate_taxi(CTaxiEnv *env)
{
	env->observations = (float *)calloc(env->num_agents * 4, sizeof(float));
	env->actions = (int *)calloc(env->num_agents, sizeof(int));
	env->rewards = (float *)calloc(env->num_agents, sizeof(float));
	env->terminals =
	    (unsigned char *)calloc(env->num_agents, sizeof(unsigned char));
	env->client = NULL;
}

void reset_taxi(CTaxiEnv *env)
{
	env->current_step = 0;
	env->taxi_row = rand() % 5;
	env->taxi_col = rand() % 5;
	env->pass_idx = rand() % 4; // Passenger at R, G, B, or Y
	env->dest_idx = rand() % 4; // Dest at R, G, B, or Y

	while (env->pass_idx == env->dest_idx) {
		env->dest_idx = rand() % 4;
	}

	memset(env->terminals, 0, env->num_agents * sizeof(unsigned char));
	compute_obs(env);
}

void step_taxi(CTaxiEnv *env)
{
	int action = env->actions[0];
	float reward = -1.0f; // Default penalty for taking time
	bool done = false;

	int r = env->taxi_row;
	int c = env->taxi_col;

	if (action == 0) { // DOWN
		if (r < 4)
			env->taxi_row++;
	} else if (action == 1) { // UP
		if (r > 0)
			env->taxi_row--;
	} else if (action == 2) { // RIGHT
		// Check bounds AND wall to the right
		if (c < 4 && V_WALLS[r][c] == 0)
			env->taxi_col++;
	} else if (action == 3) { // LEFT
		// Check bounds AND wall to the right of the PREVIOUS cell
		if (c > 0 && V_WALLS[r][c - 1] == 0)
			env->taxi_col--;
	}

	else if (action == 4) {		 // PICKUP
		if (env->pass_idx < 4) { // Is passenger at a location?
			int pr = LOCS[env->pass_idx][0];
			int pc = LOCS[env->pass_idx][1];
			if (env->taxi_row == pr && env->taxi_col == pc) {
				env->pass_idx =
				    4; // Successfully picked up (now in taxi)
			} else {
				reward = -10.0f; // Illegal pickup
			}
		} else {
			reward = -10.0f; // Already have passenger
		}
	} else if (action == 5) {	  // DROPOFF
		if (env->pass_idx == 4) { // Do we have a passenger?
			int dr = LOCS[env->dest_idx][0];
			int dc = LOCS[env->dest_idx][1];
			if (env->taxi_row == dr && env->taxi_col == dc) {
				reward = 20.0f; // SUCCESS!
				done = true;
				env->pass_idx =
				    env->dest_idx; // Put passenger at dest
			} else {
				reward = -10.0f; // Wrong location
			}
		} else {
			reward = -10.0f; // No passenger to drop
		}
	}

	env->current_step++;
	if (env->current_step >= env->max_steps)
		done = true;

	env->rewards[0] = reward;
	env->terminals[0] = done ? 1 : 0;

	compute_obs(env);
}

void free_taxi(CTaxiEnv *env)
{
	if (env->observations)
		free(env->observations);
	if (env->actions)
		free(env->actions);
	if (env->rewards)
		free(env->rewards);
	if (env->terminals)
		free(env->terminals);
	if (env->client)
		free(env->client);
}
