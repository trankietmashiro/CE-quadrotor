
DYNAMICS=@quadrotor;

nX = 12;%number of states
nU = 4;%number of inputs

% quadrotor w^2 to force/torque matrix
kf = 8.55*(1e-6)*91.61;
L = 0.17;
b = 1.6*(1e-2)*91.61;
m = 0.716;
g = 9.81;

A = [kf, kf, kf, kf; ...
    0, L*kf, 0, -L*kf; ...
    -L*kf, 0, L*kf, 0; ...
    b, -b, b, -b];

%initial conditions
x0= [0;0;0;0;0;0;0;0;0;0;0;0];
xd= [10;10;10;0;0;0;0;0;0;0;0;0];

% Initialization
num_samples = 1000;
N = 150;

utraj = zeros(nU, N-1);
uOpt = [];
xf = [];
dt = 0.02;
lambda = 10;
nu = 1000;

covu0 = diag([6.25,25*1e-6,25*1e-6,25*1e-6]);
covu = repmat(covu0, 1, 1, N-1);  % 4x4xN
%covu = 1*ones(nU, N-1);
num_elites = floor(0.02*num_samples);


xtraj = zeros(nX, N);
R = lambda*inv(covu0);

x = x0;

%% Run CE Optimization
for iter = 1:500
    iter
    xf = [xf,x]; % Append the simulated trajectory
    Straj = zeros(1,num_samples); % Initialize cost of rollouts
    
    % Start the rollouts and compute rollout costs
    for k = 1:num_samples
        xtraj = [];
        xtraj(:,1) = x;
        for t = 1:N-1
            du = sqrt(covu(:,:,t)) * randn(nU,1);
            dU(:,t,k) = du;
            udu(:,t,k) = utraj(:,t) + du;
            xtraj(:,t+1) = xtraj(:,t) + DYNAMICS(xtraj(:,t), udu(:,t,k))*dt;
            Straj(k) = Straj(k) + runningCost(xtraj(:,t), xd, R, utraj(:,t), du, nu);
        end
        Straj(k) = Straj(k) + finalCost(xtraj(:,N), xd);
    end
    
    [Se, idxe] = mink(Straj, num_elites);
    [minS, minidx] = min(Straj) % Minimum rollout cost
    
    
    % Update distribution
    umin = udu(:,:, minidx);
    udue = udu(:,:, idxe);
    
    for t = 1:N-1
    	utraj(:,t) = mean(udue(:,t,:),3);
        variances = mean((udue(:,t,:) - utraj(:,t)).^2, 3);             
        covu(:, :, t) = diag(variances);             
    end
    
    % Execute the utraj(0)
    x = x + DYNAMICS(x, utraj(:,1))*dt;
    uOpt = [uOpt, utraj(:,1)];

    % Shift utraj and covu
    for t = 2:N-1
        utraj(:,t-1) = utraj(:,t);
        covu(:,:,t-1) = covu(:,:,t);
    end
    utraj(:,N-1) = [0;0;0;0];
    covu(:,:,N-1) = diag([6.25,25*1e-6,25*1e-6,25*1e-6]);
    
    distance = norm(x(1:3)-xd(1:3)) %Current distance to target
end

%% Helper functions
function J = runningCost(x, xd, R, u, du, nu)
    Q = diag([2.5, 2.5, 20, 1, 1, 15, zeros(1, 6)]);
    qx = (x-xd)'*Q*(x-xd);
    J = qx + 1/2*u'*R*u + (1-1/nu)/2*du'*R*du + u'*R*du;
end

function J = finalCost(xT,xd)
    Qf = 20*diag([2.5, 2.5, 20, 1, 1, 15, zeros(1, 6)]);
    J = (xT-xd)'*Qf*(xT-xd);
end
