global num_of_states
global start_state
global step_range
global terminal_state
global Actions
global states

num_of_states = 1000;
start_state = 500;
step_range = 100;
terminal_state = [0, num_of_states + 1];
Actions = [-1,1];
states = (1:1:num_of_states);

episodes = 1*10^(5);
step_size = 2*10^(-5);
num_of_groups = 10;
W = zeros(1, num_of_groups);
distribution = zeros(1, num_of_states+2);

%% main 
true_value = compute_true_value();
for episode = 1: episodes
        [W, distribution] = gradient_monte_carlo(W, step_size, distribution);
end
approximate_values = [];
for state = states
    approximate_values = [approximate_values, apporximate_value(W,state)];
end
distribution = distribution./sum(distribution);

plot(states, true_value(2:end-1))
title('True value')
xlabel('State')
ylabel('Value scale')

figure;
plot(states, approximate_values)
title('Approximate MC value')
xlabel('State')
ylabel('Value scale')
figure;

plot(states, distribution(2:end-1))
title('State distribution')
xlabel('State')
ylabel('Distribution scale')

%% random walk gradient_mc
function [W, distribution] = gradient_monte_carlo(W, step_size, distribution)
    global terminal_state
    global start_state

    state = start_state;
    trajectory = [];
    reward = 0;
    
    while ~(state == terminal_state(1) || state == terminal_state(2))
        action = get_action();
        [next_state, reward] = step(state, action); 
        trajectory = [trajectory, state];
        state = next_state;
    end
    
    for state = trajectory
        delta_W = step_size * (reward - apporximate_value(W, state));
        W = update(W, state, delta_W);
        distribution(state+1) = distribution(state+1) + 1;
    end

end

%% get random action
function action = get_action()
    if binornd(1,0.5) == 1
        action = 1;
    else
        action = -1; 
    end
end

%% take an random action at each state. retrun reward and new state
function [state, reward] = step(state, action)    
    global num_of_states
    global step_range
    
    step = randi(step_range, 1);
    step = step*action;
    
    state = state + step;
    state = max(min(state, num_of_states + 1), 0);
    
    if state == 0
        reward = -1; 
    elseif state == num_of_states + 1
        reward = 1;   
    else
        reward = 0;
    end
end

%% get approximate value of state
function val = apporximate_value(W, state)
    global terminal_state
    global step_range
    
    if state == terminal_state(1) || state == terminal_state(2)
        val = 0; 
    else
        group_num = ceil(state/step_range);
        val = W(group_num); 
    end           
end

%% update parameters       
function W = update(W, state, delta_W)
    global step_range
    group_num = ceil(state/step_range);
    W(group_num) = W(group_num) + delta_W;            
end

%% compute true value
function true_value_ = compute_true_value()
    global num_of_states
    global step_range
    global states
    global Actions  
    true_value_ = (-1001 : 2 : 1001)/1001;  
    while(true)          
        old_value = true_value_;   
        for state = 1:length(states)
            true_value_(state+1) = 0;
            for action = Actions
                for step = 1 : step_range
                    step_ = step * action;
                    next_state = (state+1) + step_;
                    next_state = max(min(next_state, num_of_states + 2), 1);            
                    true_value_(state+1) = true_value_(state+1) + (1 / (2 * 100)) * true_value_(next_state);
                end
            end 
        end
        error = sum(abs(old_value - true_value_));
        if error < 0.01
            break
        end
    end
        true_value_(1) = 0;
        true_value_(end) = 0;
end
