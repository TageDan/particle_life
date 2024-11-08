use macroquad::color::Color;
use macroquad::ui::{hash, root_ui, widgets};
use miniquad::EventHandler;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};
use std::io::stdout;
use std::ops::{Add, Mul};
use std::sync::{Arc, Mutex};
use std::thread;
// use termion;

use macroquad::{prelude::*, rand};
// use termion::raw::IntoRawMode;

#[derive(Clone, Copy)]
struct Point {
    x: f32,
    y: f32,
}

const MAX_TYPES: u32 = 8;
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
}

#[derive(Clone)]
struct Particle {
    pos: Point,
    vel: Point,
    particle_type: u32,
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
    types: u32,
    particle_amount: u32,
    variables: &VariableState,
) -> (Vec<Particle>, Arc<[Vec<f32>]>) {
    let particles = generate_particles(types, particle_amount, variables);

    let rules = generate_random_rules(types);
    return (particles, rules);
}

fn generate_random_rules(types: u32) -> Arc<[Vec<f32>]> {
    let mut rules = Vec::new();
    for _ in 0..types {
        let mut rule_row = Vec::new();
        for _ in 0..types {
            let r = rand::gen_range(0.0, 1.0_f32) * 2. - 1.;
            // let r: f32 = 0.;
            rule_row.push(r);
        }
        rules.push(rule_row);
    }
    rules.into()
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
    types: u32,
    particle_amount: u32,
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
        window_width: 1000,
        window_height: 800,
        ..Default::default()
    }
}

fn generate_colors(types: u32) -> Vec<RgbColor> {
    let mut colors = Vec::new();
    for _ in 0..types {
        colors.push(RgbColor(
            rand::gen_range(0, 255),
            rand::gen_range(0, 255),
            rand::gen_range(0, 255),
        ));
    }
    colors
}

fn get_forces(
    rules: Arc<[Vec<f32>]>,
    current_particle_positions: &Vec<Particle>,
    variables: &VariableState,
) -> Vec<Point> {
    let grid = get_grid(&current_particle_positions, variables);
    let variables = Arc::new(variables.clone());
    let screen_width = (variables.grid_size * variables.cols) as f32;
    let screen_height = (variables.grid_size * variables.rows) as f32;
    let cols = variables.cols;
    let rows = variables.rows;

    let forces: Vec<_> = (0..(cols as usize * rows as usize))
        .into_par_iter()
        .map(|idx| {
            let row = idx / variables.cols as usize;
            let col = idx % variables.cols as usize;
            let compare_particles =
                get_particles_to_compare(col, row, cols as usize, rows as usize, &grid);
            let mut grid_force = Vec::with_capacity(grid[col][row].len());
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
                    } = &current_particle_positions[*particle_index];
                    let Particle {
                        pos: other_pos,
                        vel: _,
                        particle_type: other_col,
                    } = &current_particle_positions[*other_particle];
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
                        rules[*this_col as usize][*other_col as usize],
                        Point { x: dx, y: dy },
                        &variables,
                    );
                    this_force = this_force + added_force;
                }
                grid_force.push(this_force);
            }
            return grid_force;
        })
        .collect();
    let mut final_forces = vec![Point { x: 0., y: 0. }; variables.particle_amount as usize];
    for (c, col) in grid.iter().enumerate() {
        for (r, row) in col.iter().enumerate() {
            for (i, idx) in row.iter().enumerate() {
                final_forces[*idx] = forces[r * cols as usize + c][i];
            }
        }
    }
    return final_forces;
}

fn get_particles_to_compare(
    col: usize,
    row: usize,
    cols: usize,
    rows: usize,
    grid: &[Vec<Vec<usize>>],
) -> Vec<usize> {
    let mut particles_to_compare = Vec::with_capacity(10);
    let col = col as i32;
    let row = row as i32;
    for i in -1_i32..2 {
        for j in -1_i32..2 {
            if col + i < 0 && row + j < 0 {
                particles_to_compare.push(&grid[cols - 1][rows - 1])
            }
            if col + i < 0 {
                particles_to_compare.push(&grid[cols - 1][(row + j) as usize % rows])
            } else if row + j < 0 {
                particles_to_compare.push(&grid[(col + i) as usize % cols][rows - 1])
            } else {
                particles_to_compare
                    .push(&grid[(col + i) as usize % cols][(row + j) as usize % rows])
            }
        }
    }
    particles_to_compare
        .iter()
        .flat_map(|x| x.iter())
        .map(|x| *x)
        .collect()
}

fn get_grid(particles: &[Particle], variables: &VariableState) -> Arc<[Vec<Vec<usize>>]> {
    let mut grids: Vec<Vec<Vec<usize>>> = Vec::with_capacity(variables.cols as usize);
    for _ in 0..variables.cols {
        let mut col = Vec::with_capacity(variables.rows as usize);
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
    grids.into()
}

struct Camera {
    x: f32,
    y: f32,
    zoom: f32,
}

impl Camera {
    fn draw(
        &self,
        particles: &mut Vec<Particle>,
        colors: &mut Vec<RgbColor>,
        variables: &mut VariableState,
        rules: &mut Arc<[Vec<f32>]>,
    ) {
        clear_background(GRAY);

        widgets::Window::new(hash!(), Vec2::splat(0.), Vec2::new(300., 400.))
            .label("Particle life")
            .titlebar(true)
            .movable(true)
            .ui(&mut *root_ui(), |ui| {
                ui.checkbox(hash!(), "running", &mut variables.running);
                if ui.button(None, "randomize colors") {
                    *colors = generate_colors(variables.types);
                }
                if ui.button(None, "regenerate particles") {
                    *particles =
                        generate_particles(variables.types, variables.particle_amount, &variables);
                }
                if ui.button(None, "randomize rules") {
                    *rules = generate_random_rules(variables.types);
                }
                ui.tree_node(hash!(), "general", |ui| {
                    let mut particle_amount = variables.particle_amount as f32;
                    ui.slider(hash!(), "particles: ", 0.0..20_000.0, &mut particle_amount);
                    let mut types = variables.types as f32;
                    ui.slider(hash!(), "types: ", 1.0..(MAX_TYPES as f32), &mut types);
                    if variables.types != types as u32 {
                        variables.types = types as u32;
                        *rules = generate_random_rules(variables.types);
                        *colors = generate_colors(variables.types);
                        *particles = generate_particles(
                            variables.types,
                            variables.particle_amount,
                            &variables,
                        );
                    }

                    if variables.particle_amount != particle_amount as u32 {
                        variables.particle_amount = particle_amount as u32;
                        *particles = generate_particles(
                            variables.types,
                            variables.particle_amount,
                            &variables,
                        );
                    }
                });
                ui.tree_node(hash!(), "physics", |ui| {
                    ui.slider(hash!(), "max_force", 0_f32..500., &mut variables.max_force);
                    ui.slider(
                        hash!(),
                        "max_radius",
                        0_f32..(variables.grid_size as f32),
                        &mut variables.max_radius,
                    );

                    ui.slider(
                        hash!(),
                        "delta_time",
                        0.005_f32..0.05,
                        &mut variables.delta_time,
                    );
                    ui.slider(hash!(), "friction", 0_f32..10.0, &mut variables.friction);
                    ui.slider(
                        hash!(),
                        "repell threshold: ",
                        0_f32..(variables.grid_size as f32 / 2.),
                        &mut variables.repell_threshold,
                    );
                });
                ui.tree_node(hash!(), "world", |ui| {
                    let mut grid_size = variables.grid_size as f32;
                    ui.slider(hash!(), "grid_size: ", 0.0..100.0, &mut grid_size);
                    variables.grid_size = grid_size as u32;
                    let mut rows = variables.rows as f32;
                    ui.slider(hash!(), "rows / cols ", 0.0..25.0, &mut rows);
                    variables.rows = rows as u32;
                    variables.cols = variables.rows;
                });

                ui.tree_node(hash!(), "types and rules", |ui| {
                    for t1 in 0..variables.types {
                        ui.tree_node(hash!(&format!("{}", t1)), &format!("type {}:", t1), |ui| {
                            let color = colors[t1 as usize].clone();
                            let mut red = color.0 as f32;
                            let mut blue = color.1 as f32;
                            let mut green = color.2 as f32;
                            ui.slider(hash!(&format!("{}r", t1)), "r", 0.0..256.0, &mut red);
                            ui.slider(hash!(&format!("{}b", t1)), "b", 0.0..256.0, &mut blue);
                            ui.slider(hash!(&format!("{}g", t1)), "g", 0.0..256.0, &mut green);
                            colors[t1 as usize] = RgbColor(red as u8, blue as u8, green as u8);
                            for t2 in 0..variables.types {
                                let mut attraction = rules[t1 as usize][t2 as usize];
                                ui.slider(
                                    hash!(&format!("{}{}", t1, t2)),
                                    &format!("attraction to type {}", t2),
                                    -1.0..1.0,
                                    &mut attraction,
                                );
                                let mut new_rules = (*rules).clone().to_vec();
                                new_rules[t1 as usize][t2 as usize] = attraction;
                                *rules = new_rules.into();
                            }
                        });
                    }
                });

                ui.label(None, "Camera Instructions:");
                ui.label(None, "WASD to move");
                ui.label(None, "UP/DOWN arrow keys to zoom in/out");
            });

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
        } in particles.iter()
        {
            draw_circle(
                (pos.x - (variables.grid_size * variables.cols) as f32 / 2. - self.x) * self.zoom
                    + screen_width() / 2.,
                (pos.y - (variables.grid_size * variables.rows) as f32 / 2. - self.y) * self.zoom
                    + screen_height() / 2.,
                (RADIUS * self.zoom).max(0.75),
                colors[*particle_type as usize].to_macroquad(),
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
    particle_amount: u32,
    types: u32,
    running: bool,
    max_force: f32,
    grid_size: u32,
    rows: u32,
    cols: u32,
    max_radius: f32,
    delta_time: f32,
    friction: f32,
    repell_threshold: f32,
}

#[macroquad::main(window_conf)]
async fn main() {
    // clear_background(BLACK);
    // let _stdout = stdout().into_raw_mode().unwrap();
    let mut variables = VariableState {
        types: 4,
        particle_amount: 10000,
        running: false,
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
        if variables.running {
            let forces = get_forces(rules.clone(), &particles, &variables);
            particles
                .par_iter_mut()
                .enumerate()
                .for_each(|(idx, particle)| {
                    particle.update(forces[idx], &variables);
                });
        }
        if macroquad::input::is_key_down(KeyCode::Down) {
            if camera.zoom > 0.5 + 0.2 {
                camera.zoom -= 0.2;
            } else {
                camera.zoom = 0.5;
            }
        }
        if macroquad::input::is_key_down(KeyCode::Up) {
            camera.zoom += 0.2;
        }
        if macroquad::input::is_key_down(KeyCode::A) {
            camera.x -= 1.0 * 10. / camera.zoom;
        }
        if macroquad::input::is_key_down(KeyCode::W) {
            camera.y -= 1.0 * 10. / camera.zoom;
        }
        if macroquad::input::is_key_down(KeyCode::S) {
            camera.y += 1.0 * 10. / camera.zoom;
        }
        if macroquad::input::is_key_down(KeyCode::D) {
            camera.x += 1.0 * 10. / camera.zoom;
        }

        camera.draw(&mut particles, &mut colors, &mut variables, &mut rules);
        next_frame().await
    }
}
