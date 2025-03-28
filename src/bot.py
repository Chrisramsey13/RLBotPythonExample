from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.messages.flat.QuickChatSelection import QuickChatSelection
from rlbot.utils.structures.game_data_struct import GameTickPacket
import math
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
    
def get_enemy_goal_location(team):
    if team == 0:
        return Vec3(0, 5120, 0)
    else:
        return Vec3(0, -5120, 0)

def calculate_azimuth_to_ball(car_location, car_rotation, ball_location):
    """
    Calculates the azimuth (horizontal angle) from the car to the ball in degrees.

    Args:
        car_location (Vec3): The location of the car.
        car_rotation (Rotation): The rotation of the car (pitch, yaw, roll).
        ball_location (Vec3): The location of the ball.

    Returns:
        float: The azimuth angle in degrees. Positive means the ball is to the right, negative means to the left.
    """
    # Get the direction to the ball
    direction_to_ball = (ball_location - car_location).normalized()

    # Calculate the car's forward direction based on its yaw
    car_forward = Vec3(math.cos(car_rotation.yaw), math.sin(car_rotation.yaw), 0).normalized()

    # Calculate the dot product and cross product
    dot_product = car_forward.dot(direction_to_ball)
    cross_product = car_forward.x * direction_to_ball.y - car_forward.y * direction_to_ball.x

    # Calculate the azimuth angle in radians
    azimuth_radians = math.atan2(cross_product, dot_product)

    # Convert the azimuth angle to degrees
    azimuth_degrees = math.degrees(azimuth_radians)

    return azimuth_degrees

def classify_ball_location(ball_location, team):
    
    if team == 0:
        # Blue team: positive Y is opponent's side, negative Y is team's side
        if ball_location.y > 0:
            ball_side = 1
            return "opponent side"
        else:
            ball_side = 0
            return "team side"
    else:
        # Orange team: negative Y is opponent's side, positive Y is team's side
        if ball_location.y < 0:
            ball_side = 1
            return "opponent side"
        else:
            ball_side = 0
            return "team side"
        
class possible_actions:

    def __init__(self, bot):
        self.bot = bot
        self.active_sequence = None
        controls = SimpleControllerState()

        # Return the controls associated with the beginning of the sequence so we can start right away.    
    def begin_front_flip(self, packet):
        """
        Begins a front flip maneuver.

        Args:
            packet (GameTickPacket): The game tick packet containing the current game state.

        Returns:
            SimpleControllerState: The controls to perform the front flip.
        """
        self.active_sequence = Sequence([
            ControlStep(duration=0.05, controls=SimpleControllerState(jump=True)),
            ControlStep(duration=0.05, controls=SimpleControllerState(jump=False)),
            ControlStep(duration=0.2, controls=SimpleControllerState(jump=True, pitch=-1)),
            ControlStep(duration=0.8, controls=SimpleControllerState()),
        ])

        return self.active_sequence.tick(packet)


'''
def get_kickoff_position(car_location):
    """
    Determines which kickoff position the car is in based on its location.
    
    Args:
        car_location (Vec3): The location of the car in XYZ coordinates.
    
    Returns:
        str: The name of the kickoff position.
    """''''''
    # Define known kickoff positions
    kickoff_positions = {
        "center": [Vec3(0, 4608, 17), Vec3(0, -4608, 17)],
        "left": [Vec3(-2048, 2560, 17), Vec3(2048, -2560, 17)],
        "right": [Vec3(2048, 2560, 17), Vec3(-2048, -2560, 17)],
        "back_left": [Vec3(-256, 3840, 17), Vec3(256, -3840, 17)],
        "back_right": [Vec3(256, 3840, 17), Vec3(-256, -3840, 17)]
    }
    
    # Determine which position the car is in
    for position_name, positions in kickoff_positions.items():
        for position_coords in positions:
            if car_location.dist(position_coords) < 100:  # Adjust the threshold as needed
                return position_name
    
    return "unknown"
'''
class MyBot(BaseAgent):

    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.active_sequence: Sequence = None
        self.boost_pad_tracker = BoostPadTracker()
        self.moves = possible_actions(self)  # Pass 'self' to the constructor
    
    def begin_front_flip(self, packet):
        """
        Begins a front flip maneuver.

        Args:
            packet (GameTickPacket): The game tick packet containing the current game state.

        Returns:
            SimpleControllerState: The controls to perform the front flip.
        """
        self.active_sequence = Sequence([
            ControlStep(duration=0.05, controls=SimpleControllerState(jump=True)),
            ControlStep(duration=0.05, controls=SimpleControllerState(jump=False)),
            ControlStep(duration=0.2, controls=SimpleControllerState(jump=True, pitch=-1)),
            ControlStep(duration=0.8, controls=SimpleControllerState()),
        ])
        return self.active_sequence.tick(packet)

    def initialize_agent(self):
        # Set up information about the boost pads now that the game is active and the info is available
        self.boost_pad_tracker.initialize_boosts(self.get_field_info())

    def slow_down_if_ball_is_high(self, car_location, ball_location, car_velocity, stop_threshold=300, height_threshold=100):

        controls = SimpleControllerState()

        # Check if the ball is close in x and y
        xy_distance = math.sqrt((ball_location.x - car_location.x) ** 2 + (ball_location.y - car_location.y) ** 2)
        if xy_distance < stop_threshold and ball_location.z > height_threshold:
            # Ball is close in x and y, but high in z
            controls.throttle = -1.0 if car_velocity.length() > 100 else 0.0  # Reverse if moving, stop otherwise
            controls.boost = False  # Disable boost
            
        else:
            # Ball is not in the desired range, keep moving
            controls.throttle = 1.0
            

        return controls
    
    def is_point_near_line(self, point, line_start, line_end, threshold):
    
        line_direction = (line_end - line_start).normalized()
        point_direction = (point - line_start)
        projection_length = point_direction.dot(line_direction)
        closest_point = line_start + line_direction * projection_length
        distance_to_line = (point - closest_point).length()
        return distance_to_line <= threshold

    def get_boost_pad_along_path(self, car_location, ball_location):

        boost_pads = self.boost_pad_tracker.boost_pads
        closest_boost = None
        closest_distance = float('inf')

        for pad in boost_pads:
            if not pad.is_active:  # Skip inactive boost pads
                continue

            pad_location = Vec3(pad.location)
            # Check if the boost pad is near the line between the car and the ball
            if self.is_point_near_line(pad_location, car_location, ball_location, threshold=200):
                distance = car_location.dist(pad_location)
                if distance < closest_distance:
                    closest_boost = pad_location
                    closest_distance = distance

        return closest_boost
    
    def predict_straight_hit_goal(self, car_location, ball_location, team):
        """
        Predicts if driving straight into the ball will send it into the opponent's net.

        Args:
            car_location (Vec3): The location of the car.
            ball_location (Vec3): The location of the ball.
            team (int): The team of the car (0 for blue, 1 for orange).

        Returns:
            bool: True if the ball will go into the opponent's net, False otherwise.
        """
        # Get the opponent's goal location
        opponent_goal = get_enemy_goal_location(team)

        # Calculate the direction from the ball to the opponent's goal
        direction_to_goal = (opponent_goal - ball_location).normalized()

        # Calculate the direction from the car to the ball
        direction_to_ball = (ball_location - car_location).normalized()

        # Check if the car's direction to the ball aligns with the ball's direction to the goal
        alignment = direction_to_ball.dot(direction_to_goal)

        # If the alignment is close to 1, the ball is likely to go into the goal
        return alignment > 0.95  # Adjust the threshold as needed

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        controls = SimpleControllerState()
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

        # Gather some information about our car and the ball
        my_car = packet.game_cars[self.index]
        car_location = Vec3(my_car.physics.location)
        car_rotation = my_car.physics.rotation
        car_velocity = Vec3(my_car.physics.velocity)
        ball_location = Vec3(packet.game_ball.physics.location)  # Ensure ball_location is defined here
        ball_speed = Vec3(packet.game_ball.physics.velocity).length()

             

        azimuth_to_ball = calculate_azimuth_to_ball(car_location, car_rotation, ball_location)
        self.renderer.draw_string_2d(20, 190, 2, 2, f"Azimuth to Ball: {azimuth_to_ball:.2f}", self.renderer.white())


        # Classify the ball location
        ball_side = classify_ball_location(ball_location, self.team)

        # By default we will chase the ball, but target_location can be changed later
        target_location = ball_location

        if my_car.boost < 50 and car_location.dist(ball_location) > 2500:
            # If we're low on boost, let's grab some pads
            target_location = self.boost_pad_tracker.get_closest_boost_pad(car_location).location
        elif my_car.boost < 20 and car_location.dist(ball_location) > 2500:
            # If we're really low on boost, let's just go back to the nearest big boost
            target_location = self.boost_pad_tracker.get_closest_full_boost(car_location)
                       

        # By default we will chase the ball, but target_location can be changed later
        target_location = ball_location

        if my_car.boost < 70:
        # Find the closest boost pad along the path to the ball
            closest_boost_pad = self.get_boost_pad_along_path(car_location, ball_location)
            if closest_boost_pad is not None:
                target_location = closest_boost_pad

        if car_location.dist(ball_location) < 1500:  # Only check when the car is close to the ball
            controls = self.slow_down_if_ball_is_high(car_location, ball_location, car_velocity)
            if controls.throttle == 0.0:  # If the car is stopping, return the controls
                return controls
                


        if car_location.dist(ball_location):
            # We're far away from the ball, let's try to lead it a little bit
            ball_prediction = self.get_ball_prediction_struct()  # This can predict bounces, etc
            if ball_speed > 500:
                ball_in_future = find_slice_at_time(ball_prediction, packet.game_info.seconds_elapsed)
            if ball_speed > 1000:
                ball_in_future = find_slice_at_time(ball_prediction, packet.game_info.seconds_elapsed + 0.3)
            if ball_speed > 1500:
                ball_in_future = find_slice_at_time(ball_prediction, packet.game_info.seconds_elapsed + 0.5)  

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

        if self.predict_straight_hit_goal(car_location, ball_location, self.team):
            self.renderer.draw_string_2d(20, 260, 2, 2, "Straight Hit: Goal Likely!", self.renderer.green())
            target_location = ball_location
            controls.throttle = 1.0
            controls.steer = steer_toward_target(my_car, target_location)
            controls.boost = True
            if  car_location.dist(ball_location) < 300:
                self.begin_front_flip(packet)  # Call the front flip sequence
            return controls
        else:
            self.renderer.draw_string_2d(20, 260, 2, 2, "Straight Hit: Goal Unlikely", self.renderer.red())
            
        if packet.game_info.is_kickoff_pause:
            # If it's a kickoff, let's just drive straight towards the ball
            controls.throttle = 1.0
            controls.boost = True
            controls.steer = steer_toward_target(my_car, ball_location)
            return controls

        controls.steer = steer_toward_target(my_car, target_location)
        controls.throttle = 1.0

        if car_velocity.length() > 1000 and my_car.boost > 20 and -120 < azimuth_to_ball < 120:
            controls.boost = True
            return controls
        
    

        # Add a debug string to show the current field side of the ball
        ball_side = classify_ball_location(ball_location, self.team)  # Call the function to get the ball side
        if ball_side == "team side":
            self.renderer.draw_string_2d(20, 80, 2, 2, "Ball Side: Teamside", self.renderer.white())
        elif ball_side == "opponent side":
            self.renderer.draw_string_2d(20, 80, 2, 2, "Ball Side: Opponentside", self.renderer.white())

        return controls


