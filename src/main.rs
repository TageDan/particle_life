use macroquad::color::Color;
use std::io::stdout;
use std::ops::{Add, Mul};
use std::sync::{Arc, Mutex};
use std::thread;
use termion;

use ::rand::random;

use macroquad::{prelude::*, rand};
use termion::raw::IntoRawMode;

#[derive(Clone, Copy)]
struct Point {
    x: f32,
    y: f32,
}

const MAX_TYPES: usize = 8;
const EDITABLE_VARIABLES: usize = 8;
const CAMERA_SPEED: f32 = 5.;
const DOWN_KEY: char = '\u{f051}';
const UP_KEY: char = '\u{f052}';
const RIGHT_KEY: char = '\u{f04f}';
const LEFT_KEY: char = '\u{f050}';
const RADIUS: f32 = 1.;

impl Add<Point> for Point {
    type Output = Point;

    fn add(self, other: Point) -> Point {
        Point {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl Mul<f32> for Point {
    type Output = Point;

    fn mul(self, rhs: f32) -> Self::Output {
        Point {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

fn force_size(dist: f32, variables: &VariableState) -> f32 {
    if dist > variables.max_radius {
        return 0.;
    }
    let d = dist - variables.repell_threshold;
    let d = d.min(variables.max_radius - dist);
    return d;
}

fn force(dist: f32, rule_factor: f32, p: Point, variables: &VariableState) -> Point {
    let force_size = force_size(dist, variables);
    if force_size < 0. {
        return p
            * force_size
            * (1.0 / dist)
            * (1. / variables.repell_threshold)
            * variables.max_force;
    }
    p * force_size
        * rule_factor
        * (1.0 / dist)
        * (2. / (variables.max_radius - variables.repell_threshold))
        * variables.max_force
}

#[derive(Clone)]
struct RgbColor(u8, u8, u8);

impl RgbColor {
    fn to_macroquad(&self) -> Color {
        Color::from_rgba(self.0, self.1, self.2, 255)
    }
    fn to_termion(&self) -> termion::color::Rgb {
        termion::color::Rgb(self.0, self.1, self.2)
    }
}

#[derive(Clone)]
struct Particle {
    pos: Point,
    vel: Point,
    particle_type: usize,
}

impl Particle {
    fn update(&mut self, force: Point, variables: &VariableState) {
        self.vel = self.vel + force * variables.delta_time;
        self.vel = self.vel * (1. - variables.friction);
        self.pos = self.pos + (self.vel * variables.delta_time);
        let width = (variables.cols * variables.grid_size) as f32;
        let height = (variables.rows * variables.grid_size) as f32;
        if self.pos.x < 0. {
            self.pos.x += width;
        } else if self.pos.x >= width {
            self.pos.x -= width;
        }
        if self.pos.y >= height {
            self.pos.y -= height;
        } else if self.pos.y < 0. {
            self.pos.y += height;
        }
    }
}

fn setup(
    types: usize,
    particle_amount: usize,
    variables: &VariableState,
) -> (Vec<Particle>, Vec<Vec<f32>>) {
    let particles = generate_particles(types, particle_amount, variables);

    let rules = generate_random_rules(types);
    return (particles, rules);
}

fn generate_random_rules(types: usize) -> Vec<Vec<f32>> {
    let mut rules = Vec::new();
    for _ in 0..types {
        let mut rule_row = Vec::new();
        for _ in 0..types {
            let r = random::<f32>() * 2. - 1.;
            // let r: f32 = 0.;
            rule_row.push(r);
        }
        rules.push(rule_row);
    }
    rules
}

fn generate_zeroed_rules(types: usize) -> Vec<Vec<f32>> {
    let mut rules = Vec::new();
    for _ in 0..types {
        let mut rule_row = Vec::new();
        for _ in 0..types {
            // let r = random::<f32>() * 2. - 1.;
            let r: f32 = 0.;
            rule_row.push(r);
        }
        rules.push(rule_row);
    }
    rules
}

fn generate_particles(
    types: usize,
    particle_amount: usize,
    variables: &VariableState,
) -> Vec<Particle> {
    let mut particles = Vec::new();
    for _ in 0..particle_amount {
        particles.push(Particle {
            pos: Point {
                x: rand::gen_range(0., (variables.grid_size * variables.cols) as f32),
                y: rand::gen_range(0., (variables.grid_size * variables.rows) as f32),
            },
            vel: Point { x: 0., y: 0. },
            particle_type: rand::gen_range(0, types),
        })
    }
    particles
}

fn window_conf() -> Conf {
    Conf {
        window_title: "Particle Life".to_owned(),
        window_width: 900,
        window_height: 700,
        ..Default::default()
    }
}

fn set_background<T>(color: T) -> termion::color::Bg<T>
where
    T: termion::color::Color,
{
    termion::color::Bg(color)
}

fn print_rules_display(types: &usize, colors: &[RgbColor], rules: &Vec<Vec<f32>>) {
    print!("  ");
    for i in 0..*types {
        print!(
            "{} {}",
            set_background(colors[i].to_termion()),
            set_background(termion::color::Reset),
        );
    }
    print!("\r\n  ");
    for _ in 0..*types {
        print!("-");
    }
    print!("\r\n");
    for i in 0..*types {
        print!(
            "{} {}|",
            set_background(colors[i].to_termion()),
            set_background(termion::color::Reset),
        );
        for j in 0..*types {
            if rules[i][j] == 0. {
                print!(
                    "{}0{}",
                    set_background(RgbColor(0, 0, 0).to_termion()),
                    set_background(termion::color::Reset),
                );
            } else if rules[i][j] < 0. {
                print!(
                    "{}-{}",
                    set_background(RgbColor((rules[i][j] * 255. * -1.) as u8, 0, 0).to_termion()),
                    set_background(termion::color::Reset),
                );
            } else {
                print!(
                    "{}+{}",
                    set_background(RgbColor(0, (rules[i][j] * 255.) as u8, 0).to_termion()),
                    set_background(termion::color::Reset),
                );
            }
        }
        print!("\r\n");
    }
}

fn print_display(
    colors: &[RgbColor],
    rules: &Vec<Vec<f32>>,
    cursor_pos: &[usize],
    variables: &VariableState,
) {
    println!("{}{}", termion::clear::All, termion::cursor::Goto(1, 1));
    print_rules_display(&variables.types, colors, rules);
    cursor_goto_with_offset(&[0, 0], Some(&[0, 5 + variables.types]));
    print_variables_display(variables);
    print!("\r\n\n------------------------\r\n");
    print!("Particle Life! \r\nThis is a particle life simulation. \r\nOn the top of this menu you can see the rules of attraction between different colors. \r\nThen there are some variables that you can change for the simulation. \r\nUse the arrow keys to move around this menu.\r\nPress space to start changing the hovered value with up/down.\r\nPress space again to stop changing the value.\r\n\nCommands:\r\n(r) - run/pause\r\n(shift + r) - Generate new particles\r\n(c) - Generate new colors\r\n(+) - add new color/type\r\n(-) - remove color/type\r\n(shift + '+') - zoom in\r\n(shift + '-') - zoom out\r\n(W,A,S,D) - move camera\r\n(q)-quit");

    match variables.state {
        State::Rules => cursor_goto_with_offset(cursor_pos, Some(&[2, 2])),
        State::Variables => cursor_goto_with_offset(cursor_pos, Some(&[15, 5 + variables.types])),
    }
}

fn print_variables_display(variables: &VariableState) {
    println!(
        "{: <25}{} \r",
        "Particle Amount:", variables.particle_amount
    );
    println!("{: <25}{} \r", "Delta time:", variables.delta_time);
    println!("{: <25}{} \r", "Max force:", variables.max_force);
    println!("{: <25}{} \r", "Grids per dimension:", variables.rows);
    println!("{: <25}{} \r", "Grid size:", variables.grid_size);
    println!(
        "{: <25}{} (Maximum distance for variables to attract or repell other, recommended to be the same as Grid size)\r",
        "Max distance:", variables.max_radius
    );
    println!(
        "{: <25}{} (Particles that are closer than this distance always repell)\r",
        "Repell distance:", variables.repell_threshold
    );
    println!("{: <25}{} \r", "Friction:", variables.friction);
}

fn cursor_goto_with_offset(cursor_pos: &[usize], offset: Option<&[usize]>) {
    if let Some(margin) = offset {
        println!(
            "{}",
            termion::cursor::Goto(
                (cursor_pos[0] + 1 + margin[0]) as u16,
                (cursor_pos[1] + 1 + margin[1]) as u16
            )
        );
    } else {
        println!(
            "{}",
            termion::cursor::Goto((cursor_pos[0] + 1) as u16, (cursor_pos[1] + 1) as u16)
        );
    }
}

fn generate_colors(types: usize) -> Vec<RgbColor> {
    let mut colors = Vec::new();
    for _ in 0..types {
        colors.push(RgbColor(random::<u8>(), random::<u8>(), random::<u8>()));
    }
    colors
}

fn get_forces(
    rules: &Vec<Vec<f32>>,
    particles: &Vec<Particle>,
    variables: &VariableState,
) -> Vec<Point> {
    let rules = Arc::new(rules.clone());
    let current_particle_positions = Arc::new(particles.clone());
    let mut forces = Vec::new();
    for _ in 0..current_particle_positions.len() {
        forces.push(Point { x: 0., y: 0. });
    }
    let forces = Arc::new(Mutex::new(forces));

    let mut handles = vec![];
    let grid = Arc::new(get_grid(particles, variables));
    let variables = Arc::new(variables.clone());

    for col in 0..grid.len() {
        let forces = Arc::clone(&forces);
        let rules = Arc::clone(&rules);
        let grid = Arc::clone(&grid);
        let variables = Arc::clone(&variables);
        let variables = variables.clone();
        let current_particles = Arc::clone(&current_particle_positions);
        let screen_width = (variables.grid_size * variables.cols) as f32;
        let screen_height = (variables.grid_size * variables.rows) as f32;
        let cols = grid.len();
        let rows = grid[0].len();
        let handle = thread::spawn(move || {
            for row in 0..grid[col].len() {
                let compare_particles = get_particles_to_compare(col, row, cols, rows, &grid);
                for particle_index in &grid[col][row] {
                    let mut this_force = Point { x: 0., y: 0. };
                    for other_particle in &compare_particles {
                        if *particle_index == *other_particle {
                            continue;
                        }
                        let Particle {
                            pos: this_pos,
                            vel: _,
                            particle_type: this_col,
                        } = &current_particles[*particle_index];
                        let Particle {
                            pos: other_pos,
                            vel: _,
                            particle_type: other_col,
                        } = &current_particles[*other_particle];
                        let Point {
                            x: mut dx,
                            y: mut dy,
                        } = *other_pos + *this_pos * -1.;
                        if dx.abs() > screen_width / 2. {
                            dx = dx.signum() * (dx.abs() - screen_width);
                        }
                        if dy.abs() > screen_height / 2. {
                            dy = dy.signum() * (dy.abs() - screen_height);
                        }
                        let added_force = force(
                            (dx.powi(2) + dy.powi(2)).sqrt(),
                            rules[*this_col][*other_col],
                            Point { x: dx, y: dy },
                            &variables,
                        );
                        this_force = this_force + added_force;
                    }
                    forces.lock().expect("bleeeeeeeeeeee")[*particle_index] = this_force;
                }
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().expect("couldnt join handles");
    }
    return forces.lock().unwrap().to_owned().to_vec();
}

fn get_particles_to_compare(
    col: usize,
    row: usize,
    cols: usize,
    rows: usize,
    grid: &Vec<Vec<Vec<usize>>>,
) -> Vec<usize> {
    let mut particles_to_compare = Vec::new();
    let col = col as i32;
    let row = row as i32;
    for i in -1_i32..2 {
        for j in -1_i32..2 {
            if col + i < 0 && row + j < 0 {
                particles_to_compare.append(&mut grid[cols - 1][rows - 1].clone())
            }
            if col + i < 0 {
                particles_to_compare.append(&mut grid[cols - 1][(row + j) as usize % rows].clone())
            } else if row + j < 0 {
                particles_to_compare.append(&mut grid[(col + i) as usize % cols][rows - 1].clone())
            } else {
                particles_to_compare
                    .append(&mut grid[(col + i) as usize % cols][(row + j) as usize % rows].clone())
            }
        }
    }
    particles_to_compare
}

fn get_grid(particles: &[Particle], variables: &VariableState) -> Vec<Vec<Vec<usize>>> {
    let mut grids: Vec<Vec<Vec<usize>>> = Vec::new();
    for _ in 0..variables.cols {
        let mut col = Vec::new();
        for _ in 0..variables.rows {
            col.push(Vec::new());
        }
        grids.push(col);
    }
    for (
        i,
        Particle {
            pos,
            vel: _,
            particle_type: _,
        },
    ) in particles.iter().enumerate()
    {
        let mut x = pos.x.div_euclid(variables.grid_size as f32) as usize;
        let mut y = pos.y.div_euclid(variables.grid_size as f32) as usize;
        if x >= grids.len() {
            x = grids.len() - 1;
        }
        if y >= grids[x].len() {
            y = grids[x].len() - 1;
        }
        grids[x][y].push(i);
    }
    grids
}

fn handle_input_edit_rules(
    ch: char,
    particles: &mut Vec<Particle>,
    cursor_pos: &mut [usize; 2],
    rules: &mut Vec<Vec<f32>>,
    colors: &mut Vec<RgbColor>,
    variables: &mut VariableState,
) {
    match ch {
        DOWN_KEY => {
            if !variables.editing {
                if cursor_pos[1] < variables.types - 1 {
                    cursor_pos[1] = cursor_pos[1].saturating_add(1);
                } else {
                    *cursor_pos = [0, 0];
                    variables.state = State::Variables;
                }
            } else if rules[cursor_pos[1]][cursor_pos[0]] > -1. {
                rules[cursor_pos[1]][cursor_pos[0]] -= 0.1;
            }
        }
        UP_KEY => {
            if !variables.editing {
                if cursor_pos[1] > 0 {
                    cursor_pos[1] = cursor_pos[1].saturating_sub(1);
                }
            } else if rules[cursor_pos[1]][cursor_pos[0]] < 1. {
                rules[cursor_pos[1]][cursor_pos[0]] += 0.1;
            }
        }
        LEFT_KEY => {
            if !variables.editing && cursor_pos[0] > 0 {
                cursor_pos[0] = cursor_pos[0].saturating_sub(1);
            }
        }
        RIGHT_KEY => {
            if !variables.editing && cursor_pos[0] < variables.types - 1 {
                cursor_pos[0] = cursor_pos[0].saturating_add(1);
            }
        }
        _ => (),
    }
}

#[derive(Clone)]
enum State {
    Rules,
    Variables,
}

fn handle_input(
    variables: &mut VariableState,
    particles: &mut Vec<Particle>,
    cursor_pos: &mut [usize; 2],
    rules: &mut Vec<Vec<f32>>,
    colors: &mut Vec<RgbColor>,
    camera: &mut Camera,
) {
    while let Some(input) = get_char_pressed() {
        match input {
            '+' => {
                if variables.types >= MAX_TYPES {
                    continue;
                }
                variables.types += 1;
                colors.push(RgbColor(random::<u8>(), random::<u8>(), random::<u8>()));
                *particles =
                    generate_particles(variables.types, variables.particle_amount, &variables);
                variables.running = false;
                *rules = generate_random_rules(variables.types);
            }
            '-' => {
                if variables.types < 2 {
                    continue;
                }
                variables.types = variables.types.saturating_sub(1);
                colors.pop();
                *particles =
                    generate_particles(variables.types, variables.particle_amount, &variables);
                variables.running = false;
                *rules = generate_random_rules(variables.types);
            }

            ' ' => {
                variables.editing = !variables.editing;
            }
            'r' => {
                variables.running = !variables.running;
            }
            'R' => {
                *particles =
                    generate_particles(variables.types, variables.particle_amount, &variables);
            }
            'G' => {
                *rules = generate_random_rules(variables.types);
            }
            'g' => {
                *rules = generate_zeroed_rules(variables.types);
            }
            'c' => {
                *colors = generate_colors(variables.types);
            }
            'w' => {
                camera.y -= CAMERA_SPEED * (1. - camera.zoom / 10.);
            }
            's' => {
                camera.y += CAMERA_SPEED * (1. - camera.zoom / 10.);
            }
            'a' => {
                camera.x -= CAMERA_SPEED * (1. - camera.zoom / 10.);
            }
            'd' => {
                camera.x += CAMERA_SPEED * (1. - camera.zoom / 10.);
            }
            '_' => {
                if camera.zoom > 0.01 {
                    camera.zoom *= 0.98;
                }
            }
            '?' => {
                if camera.zoom < 5. {
                    camera.zoom *= 1.02;
                }
            }
            'q' => {
                panic!("quit");
            }
            ch => match variables.state {
                State::Rules => {
                    handle_input_edit_rules(ch, particles, cursor_pos, rules, colors, variables)
                }
                State::Variables => handle_input_edit_variables(ch, variables, cursor_pos),
            },
        }
    }
}

fn handle_input_edit_variables(
    ch: char,
    variables: &mut VariableState,
    cursor_pos: &mut [usize; 2],
) {
    match ch {
        DOWN_KEY => {
            if variables.editing {
                match cursor_pos[1] {
                    0 => variables.particle_amount = variables.particle_amount.saturating_sub(100),
                    1 => {
                        variables.delta_time -= 0.0025;
                        if variables.delta_time < 0. {
                            variables.delta_time += 0.0025;
                        }
                    }
                    2 => {
                        variables.max_force -= 10.;
                        if variables.max_force < 0. {
                            variables.max_force += 10.;
                        }
                    }
                    3 => {
                        variables.rows = variables.rows.saturating_sub(1);
                        variables.cols = variables.cols.saturating_sub(1);
                    }
                    4 => variables.grid_size = variables.grid_size.saturating_sub(10),
                    5 => {
                        variables.max_radius -= 5.;
                        if variables.max_radius <= variables.repell_threshold {
                            variables.max_radius += 5.;
                        }
                    }
                    6 => {
                        variables.repell_threshold -= 5.;
                        if variables.repell_threshold <= 0. {
                            variables.repell_threshold += 5.;
                        }
                    }
                    7 => {
                        variables.friction -= 0.05;
                        if variables.friction <= 0. {
                            variables.friction += 0.05;
                        }
                    }
                    _ => (),
                }
            } else {
                if cursor_pos[1] < EDITABLE_VARIABLES {
                    cursor_pos[1] += 1;
                }
            }
        }
        UP_KEY => {
            if variables.editing {
                match cursor_pos[1] {
                    0 => variables.particle_amount = variables.particle_amount.saturating_add(100),
                    1 => variables.delta_time += 0.0025,
                    2 => variables.max_force += 10.,
                    3 => {
                        variables.rows = variables.rows.saturating_add(1);
                        variables.cols = variables.cols.saturating_add(1);
                    }
                    4 => variables.grid_size = variables.grid_size.saturating_add(10),
                    5 => {
                        variables.max_radius += 5.;
                    }
                    6 => {
                        variables.repell_threshold += 5.;
                        if variables.max_radius <= variables.repell_threshold {
                            variables.repell_threshold -= 5.;
                        }
                    }
                    7 => variables.friction += 0.05,
                    _ => (),
                }
            } else {
                if cursor_pos[1] <= 0 {
                    variables.state = State::Rules
                } else {
                    cursor_pos[1] -= 1
                }
            }
        }
        _ => {}
    }
}

struct Camera {
    x: f32,
    y: f32,
    zoom: f32,
}

impl Camera {
    fn draw(&self, particles: &Vec<Particle>, colors: &Vec<RgbColor>, variables: &VariableState) {
        clear_background(GRAY);
        self.fill_background(
            [0., 0.],
            [
                (variables.cols * variables.grid_size) as f32,
                (variables.rows * variables.grid_size) as f32,
            ],
            &variables,
        );
        for Particle {
            pos,
            vel: _,
            particle_type,
        } in particles
        {
            draw_circle(
                (pos.x - (variables.grid_size * variables.cols) as f32 / 2. - self.x) * self.zoom
                    + screen_width() / 2.,
                (pos.y - (variables.grid_size * variables.rows) as f32 / 2. - self.y) * self.zoom
                    + screen_height() / 2.,
                (RADIUS * self.zoom).max(0.75),
                colors[*particle_type].to_macroquad(),
            )
        }
    }
    fn fill_background(
        &self,
        left_upper: [f32; 2],
        right_lower: [f32; 2],
        variables: &VariableState,
    ) {
        draw_rectangle(
            (left_upper[0] - self.x - (variables.grid_size * variables.cols) as f32 / 2.)
                * self.zoom
                + screen_width() / 2.,
            (left_upper[1] - self.y - (variables.grid_size * variables.cols) as f32 / 2.)
                * self.zoom
                + screen_height() / 2.,
            right_lower[0] * self.zoom,
            right_lower[1] * self.zoom,
            BLACK,
        )
    }
}

#[derive(Clone)]
struct VariableState {
    particle_amount: usize,
    types: usize,
    editing: bool,
    state: State,
    running: bool,
    max_force: f32,
    grid_size: i32,
    rows: i32,
    cols: i32,
    max_radius: f32,
    delta_time: f32,
    friction: f32,
    repell_threshold: f32,
}

#[macroquad::main(window_conf)]
async fn main() {
    // clear_background(BLACK);
    let _stdout = stdout().into_raw_mode().unwrap();
    let mut cursor_pos: [usize; 2] = [0, 0];
    let mut variables = VariableState {
        editing: false,
        types: 4,
        particle_amount: 10000,
        running: false,
        state: State::Rules,
        max_force: 400.,
        grid_size: 70,
        rows: 6,
        cols: 6,
        max_radius: 70.,
        delta_time: 0.005,
        friction: 0.25,
        repell_threshold: 20.,
    };
    let (mut particles, mut rules) = setup(variables.types, variables.particle_amount, &variables);
    let mut colors = generate_colors(variables.types);
    let mut camera = Camera {
        x: 0.,
        y: 0.,
        zoom: 1.,
    };
    loop {
        if rules.len() < variables.types {
            rules.push(vec![0.; variables.types]);
            for rule_row in rules.iter_mut() {
                while rule_row.len() < variables.types {
                    rule_row.push(0.);
                }
            }
        }
        handle_input(
            &mut variables,
            &mut particles,
            &mut cursor_pos,
            &mut rules,
            &mut colors,
            &mut camera,
        );
        print_display(&colors, &rules, &cursor_pos, &variables);

        let forces: Vec<Point>;
        if variables.running {
            forces = get_forces(&rules, &particles, &variables);
        } else {
            forces = vec![Point { x: 0., y: 0. }; particles.len()];
        }
        for particle in 0..particles.len() {
            particles[particle].update(forces[particle], &variables);
        }
        camera.draw(&particles, &colors, &variables);
        next_frame().await
    }
}
