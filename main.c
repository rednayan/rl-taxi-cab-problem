#include "taxi.h"
#include <float.h>
#include <raylib.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// --- HYPERPARAMETERS ---
#define NUM_EPISODES 5000
#define NUM_STATES 500
#define NUM_ACTIONS 6
#define ALPHA 0.1f
#define GAMMA 0.99f
#define EPSILON 0.1f

#define LOG_ALL 0
#define LOG_EDGE_CASES 5

#define GRAPH_WIDTH 1280
#define EPISODES_PER_POINT (NUM_EPISODES / GRAPH_WIDTH)

float q_table[NUM_STATES][NUM_ACTIONS];
float reward_history[GRAPH_WIDTH];

const char *get_action_name(int action)
{
	switch (action) {
	case 0:
		return "DOWN";
	case 1:
		return "UP";
	case 2:
		return "RIGHT";
	case 3:
		return "LEFT";
	case 4:
		return "PICKUP";
	case 5:
		return "DROP";
	default:
		return "UNKNOWN";
	}
}

void log_step(int episode, int step, int state_idx, int action, float reward,
	      int next_state_idx, float old_q, float new_q, CTaxiEnv *env)
{
	printf("[EP %d | ST %d] ", episode, step);
	printf("Pos:(%d,%d) Pass:%d Dest:%d", env->taxi_row, env->taxi_col,
	       env->pass_idx, env->dest_idx);
	printf(" || ACT: %-6s", get_action_name(action));
	printf(" || REW: %+.0f", reward);
	// Show the math
	printf(" || Q[%d] Update: %.4f -> %.4f\n", state_idx, old_q, new_q);
}

typedef struct {
	int cell_size;
} RenderClient;

void render_taxi(CTaxiEnv *env)
{
	if (env->client == NULL) {
		RenderClient *rc = (RenderClient *)malloc(sizeof(RenderClient));
		rc->cell_size = 100;
		env->client = rc;
	}
	RenderClient *rc = (RenderClient *)env->client;

	BeginDrawing();
	ClearBackground(RAYWHITE);

	for (int i = 0; i <= 5; ++i) {
		DrawLine(0, i * rc->cell_size, 500, i * rc->cell_size,
			 LIGHTGRAY);
		DrawLine(i * rc->cell_size, 0, i * rc->cell_size, 500,
			 LIGHTGRAY);
	}

	// Draw walls (V_WALLS[r][c] == 1 means wall to the right of cell)
	static const int V_WALLS[5][5] = {{0, 1, 0, 0, 0},
					  {0, 1, 0, 0, 0},
					  {0, 0, 0, 0, 0},
					  {0, 1, 0, 1, 0},
					  {0, 1, 0, 1, 0}};
	for (int r = 0; r < 5; ++r) {
		for (int c = 0; c < 5; ++c) {
			if (V_WALLS[r][c] == 1) {
				int x = (c + 1) * rc->cell_size;
				int y1 = r * rc->cell_size;
				int y2 = (r + 1) * rc->cell_size;
				DrawLineEx((Vector2){(float)x, (float)y1},
					   (Vector2){(float)x, (float)y2}, 4.0f,
					   BLACK);
			}
		}
	}

	Color colors[4] = {RED, GREEN, YELLOW, BLUE};
	const char *labels[4] = {"R", "G", "Y", "B"};
	for (int i = 0; i < 4; ++i) {
		int r = LOCS[i][0];
		int c = LOCS[i][1];
		DrawRectangle(c * rc->cell_size + 5, r * rc->cell_size + 5, 30,
			      30, colors[i]);
		DrawText(labels[i], c * rc->cell_size + 12,
			 r * rc->cell_size + 10, 20, WHITE);
	}

	int dr = LOCS[env->dest_idx][0];
	int dc = LOCS[env->dest_idx][1];
	DrawRectangleLines(dc * rc->cell_size + 2, dr * rc->cell_size + 2,
			   rc->cell_size - 4, rc->cell_size - 4, PURPLE);

	if (env->pass_idx < 4) {
		int r = LOCS[env->pass_idx][0];
		int c = LOCS[env->pass_idx][1];
		DrawCircle(c * rc->cell_size + 50, r * rc->cell_size + 50, 15,
			   PURPLE);
	}

	Color taxiColor = (env->pass_idx == 4) ? GREEN : ORANGE;
	DrawRectangle(env->taxi_col * rc->cell_size + 20,
		      env->taxi_row * rc->cell_size + 20, 60, 60, taxiColor);

	DrawText("AGENT", 10, 510, 20, DARKGRAY);
	DrawText(TextFormat("Reward: %.1f", env->rewards[0]), 300, 510, 20,
		 BLACK);
	EndDrawing();
}

void show_learning_graph()
{
	int window_width = 1280;
	int window_height = 720;

	int left_margin = 60;
	int bottom_margin = 50;
	int top_margin = 50;
	int right_margin = 20;

	int plot_width = window_width - left_margin - right_margin;
	int plot_height = window_height - top_margin - bottom_margin;

	InitWindow(window_width, window_height, "Learning Curve");
	SetTargetFPS(60);

	float max_val = -FLT_MAX;
	float min_val = FLT_MAX;
	for (int i = 0; i < GRAPH_WIDTH; i++) {
		if (reward_history[i] > max_val)
			max_val = reward_history[i];
		if (reward_history[i] < min_val)
			min_val = reward_history[i];
	}

	float range = max_val - min_val;
	if (range == 0)
		range = 1;
	max_val += range * 0.05f;
	min_val -= range * 0.05f;
	range = max_val - min_val;

	float smoothed[GRAPH_WIDTH];
	int window_size = 15;
	for (int i = 0; i < GRAPH_WIDTH; i++) {
		float sum = 0;
		int count = 0;
		for (int j = i - window_size / 2; j <= i + window_size / 2;
		     j++) {
			if (j >= 0 && j < GRAPH_WIDTH) {
				sum += reward_history[j];
				count++;
			}
		}
		smoothed[i] = sum / count;
	}

	while (!WindowShouldClose()) {
		if (IsKeyPressed(KEY_SPACE))
			break;

		BeginDrawing();
		ClearBackground(RAYWHITE);

		DrawText("TRAINING PERFORMANCE", left_margin, 10, 20, DARKGRAY);
		DrawText("[INFO]Training Complete: Press SPACE to Start the "
			 "Simulation",
			 left_margin, window_height - 25, 20, DARKGREEN);

		for (int i = 0; i <= 5; i++) {
			float normalized = (float)i / 5.0f;
			int y = top_margin + plot_height -
				(int)(normalized * plot_height);
			float value = min_val + normalized * range;

			DrawLine(left_margin, y, window_width - right_margin, y,
				 Fade(LIGHTGRAY, 0.5f));
			DrawText(TextFormat("%.0f", value), 5, y - 5, 10,
				 DARKGRAY);
		}

		for (int i = 0; i <= 5; i++) {
			float normalized = (float)i / 5.0f;
			int x = left_margin + (int)(normalized * plot_width);
			int episode_num = (int)(normalized * NUM_EPISODES);

			DrawLine(x, top_margin, x, top_margin + plot_height,
				 Fade(LIGHTGRAY, 0.5f));
			DrawText(TextFormat("%d", episode_num), x - 15,
				 top_margin + plot_height + 5, 10, DARKGRAY);
		}

		DrawText("Episodes ->", window_width / 2, window_height - 20,
			 10, DARKGRAY);

		if (min_val < 0 && max_val > 0) {
			float zero_ratio = (0 - min_val) / range;
			int zero_y = top_margin + plot_height -
				     (int)(zero_ratio * plot_height);
			DrawLine(left_margin, zero_y,
				 window_width - right_margin, zero_y, BLACK);
		}

		for (int i = 0; i < GRAPH_WIDTH - 1; i++) {
			int x1 = left_margin +
				 (int)((float)i / GRAPH_WIDTH * plot_width);
			int y1 = top_margin + plot_height -
				 (int)((reward_history[i] - min_val) / range *
				       plot_height);

			int x2 = left_margin + (int)((float)(i + 1) /
						     GRAPH_WIDTH * plot_width);
			int y2 = top_margin + plot_height -
				 (int)((reward_history[i + 1] - min_val) /
				       range * plot_height);

			DrawLine(x1, y1, x2, y2,
				 Fade(RED, 0.25f)); // High transparency
		}

		for (int i = 0; i < GRAPH_WIDTH - 1; i++) {
			int x1 = left_margin +
				 (int)((float)i / GRAPH_WIDTH * plot_width);
			int y1 = top_margin + plot_height -
				 (int)((smoothed[i] - min_val) / range *
				       plot_height);

			int x2 = left_margin + (int)((float)(i + 1) /
						     GRAPH_WIDTH * plot_width);
			int y2 = top_margin + plot_height -
				 (int)((smoothed[i + 1] - min_val) / range *
				       plot_height);

			DrawLineEx((Vector2){(float)x1, (float)y1},
				   (Vector2){(float)x2, (float)y2}, 2.0f, RED);
		}

		DrawRectangle(window_width - 150, 10, 130, 60,
			      Fade(WHITE, 0.9f));
		DrawRectangleLines(window_width - 150, 10, 130, 60, LIGHTGRAY);

		DrawLine(window_width - 140, 30, window_width - 120, 30,
			 Fade(RED, 0.3f));
		DrawText("Raw Data", window_width - 115, 25, 10, BLACK);

		DrawLineEx((Vector2){(float)window_width - 140, 50},
			   (Vector2){(float)window_width - 120, 50}, 2.0f, RED);
		DrawText("Trend (Avg)", window_width - 115, 45, 10, BLACK);

		EndDrawing();
	}
	CloseWindow();
}

int main()
{
	srand(time(NULL));
	CTaxiEnv env;
	env.num_agents = 1;
	env.max_steps = 200;
	allocate_taxi(&env);

	for (int s = 0; s < NUM_STATES; s++)
		for (int a = 0; a < NUM_ACTIONS; a++)
			q_table[s][a] = 0.0f;

	printf("Training for %d episodes...\n", NUM_EPISODES);
	printf("Logs enabled for First %d and Last %d episodes.\n\n",
	       LOG_EDGE_CASES, LOG_EDGE_CASES);

	float batch_reward_sum = 0;
	int batch_count = 0;
	int history_idx = 0;

	for (int ep = 0; ep < NUM_EPISODES; ep++) {
		reset_taxi(&env);
		int done = 0;
		float episode_total_reward = 0;

		int should_log = (LOG_ALL || ep < LOG_EDGE_CASES ||
				  ep >= NUM_EPISODES - LOG_EDGE_CASES);
		if (should_log)
			printf("--- START EPISODE %d ---\n", ep);

		while (!done) {
			int state = encode_state(env.taxi_row, env.taxi_col,
						 env.pass_idx, env.dest_idx);
			int action;

			if ((float)rand() / RAND_MAX < EPSILON) {
				action = rand() % NUM_ACTIONS;
			} else {
				int best = 0;
				for (int a = 0; a < NUM_ACTIONS; a++) {
					if (q_table[state][a] >
					    q_table[state][best])
						best = a;
				}
				action = best;
			}

			env.actions[0] = action;
			step_taxi(&env);

			float reward = env.rewards[0];
			episode_total_reward += reward;

			int next_state =
			    encode_state(env.taxi_row, env.taxi_col,
					 env.pass_idx, env.dest_idx);

			float old_q = q_table[state][action];
			float max_future = -9999.0f;

			for (int a = 0; a < NUM_ACTIONS; a++) {
				if (q_table[next_state][a] > max_future)
					max_future = q_table[next_state][a];
			}

			if (env.terminals[0])
				max_future = 0.0f;

			float new_q =
			    old_q +
			    ALPHA * (reward + GAMMA * max_future - old_q);
			q_table[state][action] = new_q;

			if (should_log) {
				log_step(ep, env.current_step, state, action,
					 reward, next_state, old_q, new_q,
					 &env);
			}

			if (env.terminals[0])
				done = 1;
		}

		batch_reward_sum += episode_total_reward;
		batch_count++;

		if (batch_count >= EPISODES_PER_POINT) {
			if (history_idx < GRAPH_WIDTH) {
				reward_history[history_idx] =
				    batch_reward_sum / batch_count;
				history_idx++;
			}
			batch_reward_sum = 0;
			batch_count = 0;
		}
	}

	printf("\nTraining Complete! Check the Graph window.\n");

	show_learning_graph();

	InitWindow(500, 550, "Debug View");
	SetTargetFPS(2); // Slow FPS to read console logs

	reset_taxi(&env);

	while (!WindowShouldClose()) {
		if (env.terminals[0])
			reset_taxi(&env);

		int state = encode_state(env.taxi_row, env.taxi_col,
					 env.pass_idx, env.dest_idx);
		int best_action = 0;

		printf("\n[INFO] State %d Values: ", state);
		for (int a = 0; a < NUM_ACTIONS; a++) {
			printf("%s:%.2f  ", get_action_name(a),
			       q_table[state][a]);
			if (q_table[state][a] > q_table[state][best_action])
				best_action = a;
		}
		printf("--> Picking: %s", get_action_name(best_action));

		env.actions[0] = best_action;
		step_taxi(&env);
		render_taxi(&env);
	}

	CloseWindow();
	free_taxi(&env);
	return 0;
}
