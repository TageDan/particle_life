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

const DOWN_KEY: char = '\u{f051}';
const UP_KEY: char = '\u{f052}';
const RIGHT_KEY: char = '\u{f04f}';
const LEFT_KEY: char = '\u{f050}';
const PARTICLES: u32 = 4500;
const MAX_RADIUS: f32 = 150.;
const DELTA_TIME: f32 = 0.02;
const REPELL_THRESHOLD: f32 = 30.;
const FRICTION: f32 = 0.2;
const RADIUS: f32 = 1.;
const THREADS: usize = 6;

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

fn force_size(dist: f32) -> f32 {
    if dist > MAX_RADIUS {
        return 0.;
    }
    let d = dist - REPELL_THRESHOLD;
    let d = d.min(MAX_RADIUS - dist);
    return d;
}

fn force(dist: f32, rule_factor: f32, p: Point) -> Point {
    let force_size = force_size(dist);
    if force_size < 0. {
        return p * force_size * (1.0 / dist);
    }
    p * force_size * rule_factor * (1.0 / dist)
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
    fn draw(&self, colors: &[RgbColor]) {
        draw_circle(
            self.pos.x,
            self.pos.y,
            RADIUS,
            colors[self.particle_type].to_macroquad(),
        )
    }

    fn update(&mut self, force: Point) {
        self.vel = self.vel + force * DELTA_TIME;
        self.vel = self.vel * (1. - FRICTION);
        self.pos = self.pos + (self.vel * DELTA_TIME);
        if self.pos.x < 0. {
            self.pos.x += screen_width();
        } else if self.pos.x > screen_width() {
            self.pos.x -= screen_width();
        }
        if self.pos.y > screen_height() {
            self.pos.y -= screen_height();
        } else if self.pos.y < 0. {
            self.pos.y += screen_height();
        }
    }
}

fn setup(types: usize) -> (Vec<Particle>, Vec<Vec<f32>>) {
    let particles = generate_particles(types);

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

fn generate_particles(types: usize) -> Vec<Particle> {
    let mut particles = Vec::new();
    for _ in 0..PARTICLES {
        particles.push(Particle {
            pos: Point {
                x: rand::gen_range(0., screen_width()),
                y: rand::gen_range(0., screen_height()),
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

fn print_display(colors: &[RgbColor], rules: &Vec<Vec<f32>>, cursor_pos: &[usize], types: usize) {
    println!("{}{}", termion::clear::All, termion::cursor::Goto(1, 1));
    print!("  ");
    for i in 0..types {
        print!(
            "{} {}",
            set_background(colors[i].to_termion()),
            set_background(termion::color::Reset),
        );
    }
    print!("\r\n  ");
    for _ in 0..types {
        print!("-");
    }
    print!("\r\n");
    for i in 0..types {
        print!(
            "{} {}|",
            set_background(colors[i].to_termion()),
            set_background(termion::color::Reset),
        );
        for j in 0..types {
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
    cursor_goto_with_offset(cursor_pos, Some(&[2, 2]));
}

fn cursor_goto_with_offset(cursor_pos: &[usize], offset: Option<&[usize]>) {
    if let Some(margin) = offset {
        println!(
            "{}",
            termion::cursor::Goto(
                (cursor_pos[0] + 1 + margin[0]) as u16,
                (cursor_pos[1] + 1 + margin[1]) as u16
            ) // termion::cursor::Goto(20, 20)
        );
    } else {
        println!(
            "{}",
            termion::cursor::Goto((cursor_pos[0] + 1) as u16, (cursor_pos[1] + 1) as u16) // termion::cursor::Goto(20, 20)
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

fn get_forces(rules: &Vec<Vec<f32>>, particles: &Vec<Particle>) -> Vec<Point> {
    let rules = Arc::new(rules.clone());
    let current_particle_positions = Arc::new(particles.clone());
    let forces = Arc::new(Mutex::new(Vec::new()));
    for _ in 0..current_particle_positions.len() {
        forces.lock().unwrap().push(Point { x: 0., y: 0. });
    }

    let mut handles = vec![];
    let chunksize = particles.len() / THREADS;
    let mut particle_vec = Vec::new();
    let mut i = 0;
    let mut j = 0;
    for _ in 0..particles.len() {
        i += 1;
        if i == chunksize {
            particle_vec.push((j, i));
            i = 0;
            j += 1;
        }
    }
    particle_vec.push((j, i));
    particle_vec
        .into_iter()
        .for_each(|(i, particle_chunk_len)| {
            let forces = Arc::clone(&forces);
            let rules = Arc::clone(&rules);
            let start_index = chunksize * i;
            let current_particles = Arc::clone(&current_particle_positions);
            let screen_width = screen_width();
            let screen_height = screen_height();
            let handle = thread::spawn(move || {
                for i2 in 0..particle_chunk_len {
                    let mut this_force = Point { x: 0., y: 0. };
                    for other_particle in 0..current_particles.len() {
                        if start_index + i2 == other_particle {
                            continue;
                        }
                        let Particle {
                            pos: this_pos,
                            vel: _,
                            particle_type: this_col,
                        } = &current_particles[start_index + i2];
                        let Particle {
                            pos: other_pos,
                            vel: _,
                            particle_type: other_col,
                        } = &current_particles[other_particle];
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
                        );
                        this_force = this_force + added_force;
                    }
                    forces.lock().expect("bleeeeeeeeeeee")[start_index + i2] = this_force;
                }
            });
            handles.push(handle);
        });

    for handle in handles {
        handle.join().expect("couldnt join handles");
    }
    return forces.lock().unwrap().to_owned().to_vec();
}

#[macroquad::main(window_conf)]
async fn main() {
    clear_background(BLACK);
    let _stdout = stdout().into_raw_mode().unwrap();
    let mut cursor_pos: [usize; 2] = [0, 0];
    let mut editing_rule = false;
    let mut types: usize = 3;
    let (mut particles, mut rules) = setup(types);
    let mut colors = generate_colors(types);
    let mut running = false;
    loop {
        let last_frame_time = get_time();
        if let Some(input) = get_char_pressed() {
            match input {
                DOWN_KEY => {
                    if !editing_rule {
                        cursor_pos[1] = cursor_pos[1].saturating_add(1);
                    } else if rules[cursor_pos[1]][cursor_pos[0]] < 1. {
                        rules[cursor_pos[1]][cursor_pos[0]] -= 0.1;
                    }
                }
                UP_KEY => {
                    if !editing_rule {
                        cursor_pos[1] = cursor_pos[1].saturating_sub(1);
                    } else if rules[cursor_pos[1]][cursor_pos[0]] < 1. {
                        rules[cursor_pos[1]][cursor_pos[0]] += 0.1;
                    }
                }
                LEFT_KEY => {
                    if !editing_rule {
                        cursor_pos[0] = cursor_pos[0].saturating_sub(1);
                    }
                }
                RIGHT_KEY => {
                    if !editing_rule {
                        cursor_pos[0] = cursor_pos[0].saturating_add(1);
                    }
                }
                '+' => {
                    types += 1;
                    colors.push(RgbColor(random::<u8>(), random::<u8>(), random::<u8>()));
                    particles = generate_particles(types);
                    running = false;
                }
                '-' => {
                    types = types.saturating_sub(1);
                    colors.pop();
                    particles = generate_particles(types);
                    running = false;
                }

                ' ' => {
                    editing_rule = !editing_rule;
                }
                'r' => {
                    running = !running;
                }
                'R' => {
                    particles = generate_particles(types);
                }
                'G' => {
                    rules = generate_random_rules(types);
                }
                'g' => {
                    rules = generate_zeroed_rules(types);
                }
                'c' => {
                    colors = generate_colors(types);
                }
                'q' => {
                    panic!("quit");
                }
                _ => {}
            }
        }
        if rules.len() < types {
            rules.push(vec![0.; types]);
            for rule_row in rules.iter_mut() {
                while rule_row.len() < types {
                    rule_row.push(0.);
                }
            }
        }
        print_display(&colors, &rules, &cursor_pos, types);
        clear_background(BLACK);

        let forces: Vec<Point>;
        if running {
            forces = get_forces(&rules, &particles);
        } else {
            forces = vec![Point { x: 0., y: 0. }; particles.len()];
        }
        for particle in 0..particles.len() {
            particles[particle].update(forces[particle]);
            particles[particle].draw(&colors);
        }
        while get_time() - last_frame_time < 1. / 30. {
            thread::sleep_ms(10);
        }
        next_frame().await
    }
}
