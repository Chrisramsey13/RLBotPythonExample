from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.messages.flat.QuickChatSelection import QuickChatSelection
from rlbot.utils.structures.game_data_struct import GameTickPacket

from util.ball_prediction_analysis import find_slice_at_time
from util.boost_pad_tracker import BoostPadTracker
from util.drive import steer_toward_target
from util.sequence import Sequence, ControlStep
from util.vec import Vec3

#BoostPadTracker class is used to keep track of the status of all the boost pads on the field and place them in a table array.
class BoostPadTracker:
    def __init__(self):
        self.boost_pads = []

    def initialize_boosts(self, field_info):
        self.boost_pads = []
        for i in range(field_info.num_boosts):
            self.boost_pads.append(field_info.boost_pads[i])

    def update_boost_status(self, packet):
        for i in range(len(self.boost_pads)):
            self.boost_pads[i].is_active = packet.game_boosts[i].is_active

    def get_full_boost_locations(self):
        return [Vec3(pad.location) for pad in self.boost_pads if pad.is_full_boost]
    
    def get_closest_boost_pad(self, car):
        return min(self.boost_pads, key=lambda pad: car.dist(Vec3(pad.location)))
    
    def get_closest_full_boost(self, car):

        full_boosts = self.get_full_boost_locations()
        if len(full_boosts) == 0:
            return None
        return min(full_boosts, key=lambda pad: car.dist(pad))
     
def copy_player_inputs(packet, player_index):
    """
    Copies the inputs from a player in the game.
    
    Args:
        packet (GameTickPacket): The game tick packet containing the current game state.
        player_index (int): The index of the player whose inputs are to be copied.
    
    Returns:
        SimpleControllerState: The copied inputs from the player.
    """
    player = packet.game_cars[player_index]
    control_steps = []

    # Example: Create 10 control steps with the player's current inputs    
    '''for _ in range(10):
    
        controls = SimpleControllerState(
            throttle=player.input.throttle,
            steer=player.input.steer,
            pitch=player.input.pitch,
            yaw=player.input.yaw,
            roll=player.input.roll,
            jump=player.input.jump,
            boost=player.input.boost,
            handbrake=player.input.handbrake
        )
        control_steps.append(ControlStep(duration=0.1, controls=controls))
    
    #return controls
    '''
def get_enemy_goal_location(team):
    if team == 0:
        return Vec3(0, 5120, 0)
    else:
        return Vec3(0, -5120, 0)


def classify_ball_location(ball_location, team):
    
    if team == 0:
        # Blue team: positive Y is opponent's side, negative Y is team's side
        if ball_location.y > 0:
            fieldside = 1
            return "opponent side"
        else:
            fieldside = 0
            return "team side"
    else:
        # Orange team: negative Y is opponent's side, positive Y is team's side
        if ball_location.y < 0:
            fieldside = 1
            return "opponent side"
        else:
            fieldside = 0
            return "team side"
        
class possible_actions:
    def begin_front_flip(self, packet):

        # Do a front flip. We will be committed to this for a few seconds and the bot will ignore other
        # logic during that time because we are setting the active_sequence.
        self.active_sequence = Sequence([
            ControlStep(duration=0.05, controls=SimpleControllerState(jump=True)),
            ControlStep(duration=0.05, controls=SimpleControllerState(jump=False)),
            ControlStep(duration=0.2, controls=SimpleControllerState(jump=True, pitch=-1)),
            ControlStep(duration=0.8, controls=SimpleControllerState()),
        ])

        # Return the controls associated with the beginning of the sequence so we can start right away.
        return self.active_sequence.tick(packet)
    
        
    def begin_kickoff_center(self, packet):     
        self.active_sequence = Sequence([
            
        ])
        return self.active_sequence.tick(packet)
    
    def begin_kickoff_left(self, packet):     
        self.active_sequence = Sequence([
            
        ])

        return self.active_sequence.tick(packet)
    
    def begin_kickoff_right(self, packet):     
        self.active_sequence = Sequence([
            
        ])

        return self.active_sequence.tick(packet)
    
    def begin_kickoff_back_left(self, packet):     
        self.active_sequence = Sequence([
            
        ])

        return self.active_sequence.tick(packet)
    
    def begin_kickoff_back_right(self, packet):     
        self.active_sequence = Sequence([
            
        ])

        return self.active_sequence.tick(packet)

def begin_speed_flip(self, packet):
    """
    Begins a speed flip maneuver.
    
    Args:
        packet (GameTickPacket): The game tick packet containing the current game state.
    
    Returns:
        SimpleControllerState: The controls to perform the speed flip.
    """
    self.active_sequence = Sequence([
        ControlStep(duration=0.1, controls=SimpleControllerState(jump=True, pitch=-1, yaw=-1, boost=True)),
        ControlStep(duration=0.1, controls=SimpleControllerState(jump=False, pitch=-1, yaw=-1, boost=True)),
        ControlStep(duration=0.1, controls=SimpleControllerState(jump=True, pitch=1, yaw=1, boost=True)),
        ControlStep(duration=0.1, controls=SimpleControllerState(jump=False, pitch=1, yaw=1, boost=True)),
        ControlStep(duration=0.1, controls=SimpleControllerState(jump=False, pitch=0, yaw=0, boost=True)),
    ])

    return self.active_sequence.tick(packet)

def get_kickoff_position(car_location):
    """
    Determines which kickoff position the car is in based on its location.
    
    Args:
        car_location (Vec3): The location of the car in XYZ coordinates.
    
    Returns:
        str: The name of the kickoff position.
    """
    # Define known kickoff positions
    kickoff_positions = {
        "center": Vec3(0, 4608, 17),
        "left": Vec3(-2048, 2560, 17),
        "right": Vec3(2048, 2560, 17),
        "back_left": Vec3(-256, 3840, 17),
        "back_right": Vec3(256, 3840, 17)
    }
    
    # Determine which position the car is in
    for position_name, position_coords in kickoff_positions.items():
        if car_location.dist(position_coords) < 100:  # Adjust the threshold as needed
            return position_name
    
    return "unknown"

class MyBot(BaseAgent):

    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.active_sequence: Sequence = None
        self.boost_pad_tracker = BoostPadTracker()

    def initialize_agent(self):
        # Set up information about the boost pads now that the game is active and the info is available
        self.boost_pad_tracker.initialize_boosts(self.get_field_info())

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        """
        This function will be called by the framework many times per second. This is where you can
        see the motion of the ball, etc. and return controls to drive your car.
        """

        # Keep our boost pad info updated with which pads are currently active
        self.boost_pad_tracker.update_boost_status(packet)

        # This is good to keep at the beginning of get_output. It will allow you to continue
        # any sequences that you may have started during a previous call to get_output.
        if self.active_sequence is not None and not self.active_sequence.done:
            controls = self.active_sequence.tick(packet)
            if controls is not None:
                return controls

        # Classify the ball location
        ball_side = classify_ball_location(ball_location, self.team)
        print(f"Ball is on the {ball_side}")

        # By default we will chase the ball, but target_location can be changed later
        target_location = ball_location

        if my_car.boost < 50:
            # If we're low on boost, let's grab some pads
            target_location = self.boost_pad_tracker.get_closest_boost_pad(car_location).location
        elif my_car.boost < 20:
            # If we're really low on boost, let's just go back to the nearest big boost
            target_location = self.boost_pad_tracker.get_closest_full_boost
            


        # Gather some information about our car and the ball
        my_car = packet.game_cars[self.index]
        car_location = Vec3(my_car.physics.location)
        car_velocity = Vec3(my_car.physics.velocity)
        ball_location = Vec3(packet.game_ball.physics.location)

        kickoff_position = get_kickoff_position(car_location)
        print(f"Kickoff position: {kickoff_position}")

        # Find and print the locations of full boost pads
        full_boost_locations = self.boost_pad_tracker.get_full_boost_locations()
        for location in full_boost_locations:
            print(f"Full Boost Pad Location: {location}")

        # By default we will chase the ball, but target_location can be changed later
        target_location = ball_location

        if my_car.boost < 50:
            # If we're low on boost, let's grab some pads
            target_location = self.boost_pad_tracker.get_closest_boost_pad(car_location).location
        elif my_car.boost < 20:
            # If we're really low on boost, let's just go back to the nearest big boost
            target_location = self.boost_pad_tracker.get_closest_full_boost(car_location)
        

        if car_location.dist(ball_location) > 1500:
            # We're far away from the ball, let's try to lead it a little bit
            ball_prediction = self.get_ball_prediction_struct()  # This can predict bounces, etc
            ball_in_future = find_slice_at_time(ball_prediction, packet.game_info.seconds_elapsed + 2)

            # ball_in_future might be None if we don't have an adequate ball prediction right now, like during
            # replays, so check it to avoid errors.
            if ball_in_future is not None:
                target_location = Vec3(ball_in_future.physics.location)
                self.renderer.draw_line_3d(ball_location, target_location, self.renderer.cyan())

        # Draw some things to help understand what the bot is thinking
        self.renderer.draw_line_3d(car_location, target_location, self.renderer.white())
        self.renderer.draw_string_3d(car_location, 1, 1, f'Speed: {car_velocity.length():.1f}', self.renderer.white())
        self.renderer.draw_rect_3d(target_location, 8, 8, True, self.renderer.cyan(), centered=True)

        #add a debug string to show data about the game
        self.renderer.draw_string_2d(20, 20, 2, 2, f"Boost: {my_car.boost}", self.renderer.white())
        self.renderer.draw_string_2d(20, 60, 2, 2, f"Ball Y: {ball_location.y}", self.renderer.white())
        
        #add a debug string to show what the active sequence is
        self.renderer.draw_string_2d(20, 100, 2, 2, f"Active Sequence: {self.active_sequence}", self.renderer.white())

        #If its a kickoff determine which kickoff position the car is in and begin that kickoff
        if GameTickPacket.game_info.is_kickoff_pause:
            kickoff_position = get_kickoff_position(car_location)
            if kickoff_position == "center":
                return self.begin_kickoff_center(packet)
            elif kickoff_position == "left":
                return self.begin_kickoff_left(packet)
            elif kickoff_position == "right":
                return self.begin_kickoff_right(packet)
            elif kickoff_position == "back_left":
                return self.begin_kickoff_back_left(packet)
            elif kickoff_position == "back_right":
                return self.begin_kickoff_back_right(packet)
            else:
                return controls
            

        #add a debug string to show the current field side of the ball
        if classify_ball_location.teamside == 0:
            self.renderer.draw_string_2d(20, 80, 2, 2,f"Ball Side: Teamside", self.renderer.white())
        elif classify_ball_location.opponentside == 1:
            self.renderer.draw_string_2d(20, 80, 2, 2,f"Ball Side: Opponentside", self.renderer.white())

        #add a debug string to show the kickoff position
        if GameTickPacket.game_info.is_kickoff_pause:
            self.renderer.draw_string_2d(10, 120, 1, 1, f"Kickoff position: {kickoff_position}", self.renderer.white())
        else:
            self.renderer.draw_string_2d(10, 120, 1, 1, "Not in kickoff", self.renderer.white())


        if 750 < car_velocity.length() < 800:
            # We'll do a front flip if the car is moving at a certain speed.
            return self.begin_speed_flip(packet)

        controls = SimpleControllerState()
        controls.steer = steer_toward_target(my_car, target_location)
        controls.throttle = 1.0
        # You can set more controls if you want, like controls.boost.

        return controls

    def begin_front_flip(self, packet):

        # Do a front flip. We will be committed to this for a few seconds and the bot will ignore other
        # logic during that time because we are setting the active_sequence.
        self.active_sequence = Sequence([
            ControlStep(duration=0.05, controls=SimpleControllerState(jump=True)),
            ControlStep(duration=0.05, controls=SimpleControllerState(jump=False)),
            ControlStep(duration=0.2, controls=SimpleControllerState(jump=True, pitch=-1)),
            ControlStep(duration=0.8, controls=SimpleControllerState()),
        ])

        # Return the controls associated with the beginning of the sequence so we can start right away.
        return self.active_sequence.tick(packet)

