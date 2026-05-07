import csv
import numpy as np
import pandas as pd
from collections import deque
# my code:
from agents.std_bb.BBController import BasalBolusController 
from utils.carb_counting import carb_estimate
# end of my code
from utils.pumpAction import Pump
from utils.core import get_env, time_in_range, custom_reward, combined_shape, linear_scaling, inverse_linear_scaling
from agents.g2p2c.core import Memory, BGPredBuffer, CGPredHorizon
from utils.statespace import StateSpace
from utils.reward_func import composite_reward


class Worker:
    def __init__(self, args, mode, patients, env_ids, seed, worker_id, device):
        self.args = args
        self.episode = 0
        self.worker_mode = mode
        self.worker_id = worker_id
        self.update_timestep = args.n_step
        self.max_test_epi_len = args.max_test_epi_len
        self.max_epi_length = args.max_epi_length
        self.calibration = args.calibration
        self.simulation_seed = seed + 100
        self.patient_name = patients[args.patient_id]
        self.env_id = str(worker_id) + '_' + env_ids[args.patient_id]
        self.env = get_env(self.args, patient_name=self.patient_name, env_id=self.env_id,
                           custom_reward=custom_reward, seed=self.simulation_seed)
        # my code:
        self.args.sampling_rate = self.env.sampling_time
        # end of my code
        self.state_space = StateSpace(self.args)
        self.pump = Pump(self.args, patient_name=self.patient_name)
        # my code:
        self.bb_controller = BasalBolusController(self.args, patient_name=self.patient_name, use_bolus=True, use_cf=False)
        # end of my code
        self.std_basal = self.pump.get_basal()
        self.memory = Memory(self.args, device)
        self.bgp_buffer = BGPredBuffer(self.args)
        self.CGPredHorizon = CGPredHorizon(self.args)
        # self.episode_history = np.zeros(combined_shape(self.max_epi_length, 13), dtype=np.float32)
        # my code:
        self.episode_history = np.zeros(combined_shape(self.max_epi_length, 15), dtype=np.float32)
        # end of my code 
        self.reinit_flag = False
        self.init_env()
        self.log1_columns = ['epi', 't', 'cgm', 'meal', 'ins', 'rew', 'rl_ins', 'mu', 'sigma',
                            #  'prob', 'state_val', 'day_hour', 'day_min']
                            # my code:
                            'prob', 'state_val', 'day_hour', 'day_min', 'uq_action_std', 'uq_gate']
                            # end of my code
        self.log2_columns = ['epi', 't', 'reward', 'normo', 'hypo', 'sev_hypo', 'hyper', 'lgbi',
                             'hgbi', 'ri', 'sev_hyper', 'aBGP_rmse', 'cBGP_rmse']
        self.save_log([self.log1_columns], '/'+self.worker_mode+'/data/logs_worker_')
        self.save_log([self.log2_columns], '/'+self.worker_mode+'/data/'+self.worker_mode+'_episode_summary_')

    def init_env(self):
        if not self.reinit_flag:
            self.episode += 1
        self.counter = 0
        self.init_state = self.env.reset()
        # my code:
        self.last_state = self.init_state
        self.last_info = None
        # end of my code 
        self.cur_state, self.feat = self.state_space.update(cgm=self.init_state.CGM, ins=0, meal=0)
        self.pump.calibrate(self.init_state)
        self.calibration_process()

    def calibration_process(self):
        self.reinit_flag, cur_cgm = False, 0
        for t in range(0, self.calibration):  # open-loop simulation for calibration period.
            state, reward, is_done, info = self.env.step(self.std_basal)
            # my code:
            self.last_state = state
            self.last_info = info
            # end of my code 
            cur_cgm = state.CGM
            self.cur_state, self.feat = self.state_space.update(cgm=state.CGM, ins=self.std_basal,
                                                                meal=info['remaining_time'], hour=self.counter,
                                                                meal_type=info['meal_type'])  # info['day_hour']
            self.reinit_flag = True if info['meal_type'] != 0 else False  # meal_type zero -> no meal
        if (cur_cgm < 110 or 130 < cur_cgm) and self.worker_mode != 'training':  # checking simulati start within normo
            self.reinit_flag = True
        if self.reinit_flag:
            self.init_env()

# my code:
    def get_human_controller_action(self):
        if self.last_info is None:
            bolus_carbs = 0
        else:
            carbs = self.last_info['meal'] * self.last_info['sample_time']
            if self.args.t_meal == 0:
                bolus_carbs = carbs
            elif self.args.t_meal == self.last_info['remaining_time']:
                bolus_carbs = self.last_info['future_carb']
            else:
                bolus_carbs = 0
            if bolus_carbs != 0:
                bolus_carbs = carb_estimate(bolus_carbs, self.last_info['day_hour'], self.patient_name,
                                            type=self.args.carb_estimation_method)
        return self.bb_controller.get_action(meal=bolus_carbs, glucose=self.last_state.CGM)
# end of my code 

    def rollout(self, policy):
        ri, alive_steps, normo, hypo, sev_hypo, hyper, lgbi, hgbi, sev_hyper = 0, 0, 0, 0, 0, 0, 0, 0, 0
        aBGpred_rmse, a_horizonBG_rmse, horizon_rmse_count = -1, 0, 0
        self.bgp_buffer.clear()
        self.CGPredHorizon.reset()
        if self.worker_mode != 'training':  # fresh env for testing
            self.init_env()
        rollout_steps = self.update_timestep if self.worker_mode == 'training' else self.max_test_epi_len

        for n_steps in range(0, rollout_steps):
            policy_step = policy.get_action(self.cur_state, self.feat)
            selected_action = policy_step['action'][0]
            rl_action, pump_action = self.pump.action(agent_action=selected_action,
                                                      prev_state=self.init_state, prev_info=None)
            # my code:
            uq_action_std = float(policy_step.get('uq_action_std',[0])[0])
            uq_gate = int(self.args.use_uq == 1 and uq_action_std > self.args.uq_action_threshold)
            if uq_gate:
                pump_action = float(np.asarray(self.get_human_controller_action()).reshape(-1)[0])
                rl_action = pump_action 
            # enf of my code 
            state, _reward, is_done, info = self.env.step(pump_action)
            # my code:
            self.last_state = state 
            self.last_info = info 
            # end of my code 
            reward = composite_reward(self.args, state=state.CGM, reward=_reward)
            self.bgp_buffer.update(policy_step['a_cgm'], policy_step['c_cgm'], state.CGM)
            # calulate the horison pred error rmse
            horizon_calc_done, err = self.CGPredHorizon.update(self.cur_state, self.feat, policy_step['action'][0],
                                                               state.CGM, policy)
            if horizon_calc_done:
                a_horizonBG_rmse += err[0]
                horizon_rmse_count += 1

            if self.worker_mode == 'training':   # store -> rollout for training
                scaled_cgm = linear_scaling(x=state.CGM, x_min=self.args.glucose_min, x_max=self.args.glucose_max)
                self.memory.store(self.cur_state, self.feat, policy_step['action'][0],
                                  reward, policy_step['state_value'], policy_step['log_prob'], scaled_cgm, self.counter)
            # update -> state.
            self.cur_state, self.feat = self.state_space.update(cgm=state.CGM, ins=pump_action,
                                                                meal=info['remaining_time'], hour=(self.counter+1),
                                                                meal_type=info['meal_type'], carbs=info['future_carb']) #info['day_hour']
            self.episode_history[self.counter] = [self.episode, self.counter, state.CGM, info['meal'] * info['sample_time'],
                                                  pump_action, reward, rl_action, policy_step['mu'][0], policy_step['std'][0],
                                                  policy_step['log_prob'][0], policy_step['state_value'][0], info['day_hour'],
                                                #   info['day_min'],
                                                    # my code:
                                                info['day_min'], uq_action_std, uq_gate]
                                                # end of my code 
            self.counter += 1
            stop_factor = (self.max_epi_length - 1) if self.worker_mode == 'training' else (self.max_test_epi_len - 1)

            criteria = state.CGM <= 40 or state.CGM >= 600 or self.counter > stop_factor
            if criteria:  # episode termination criteria.
                if self.worker_mode == 'training':
                    final_val = policy.get_final_value(self.cur_state, self.feat)
                    self.memory.finish_path(final_val)

                df = pd.DataFrame(self.episode_history[0:self.counter], columns=self.log1_columns)
                df.to_csv(self.args.experiment_dir + '/' + self.worker_mode + '/data/logs_worker_' + str(self.worker_id) + '.csv',
                          mode='a', header=False, index=False)
                alive_steps = self.counter
                aBGpred_rmse, cBGpred_rmse = self.bgp_buffer.calc_simple_rmse()
                normo, hypo, sev_hypo, hyper, lgbi, hgbi, ri, sev_hyper = time_in_range(df['cgm'], df['meal'], df['ins'],
                                                                             self.episode, self.counter, display=False)
                self.save_log([[self.episode, self.counter, df['rew'].sum(), normo, hypo, sev_hypo, hyper, lgbi,
                                hgbi, ri, sev_hyper, aBGpred_rmse, cBGpred_rmse]],
                              '/' + self.worker_mode + '/data/' + self.worker_mode + '_episode_summary_')

                if self.worker_mode == 'training':
                    self.init_env()
                else:
                    break  # stop rollout if this is a testing worker!

        aBGpred_rmse, _ = self.bgp_buffer.calc_simple_rmse()
        a_horizonBG_rmse = np.sqrt(a_horizonBG_rmse / horizon_rmse_count) if horizon_rmse_count != 0 else 0
        if self.worker_mode == 'training':
            data = self.memory.get()
        else:
            data = [ri, alive_steps, normo, hypo, sev_hypo, hyper, lgbi, hgbi, sev_hyper]
        return data, aBGpred_rmse, a_horizonBG_rmse

    def save_log(self, log_name, file_name):
        with open(self.args.experiment_dir + file_name + str(self.worker_id) + '.csv', 'a+') as f:
            csvWriter = csv.writer(f, delimiter=',')
            csvWriter.writerows(log_name)
            f.close()
