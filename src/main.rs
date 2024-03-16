use std::ops::{Add, Mul};

use ::rand::{
    distributions::{Distribution, Standard},
    random, Rng,
};
use macroquad::{prelude::*, rand};

#[derive(Clone, Copy)]
struct Point {
    x: f32,
    y: f32,
}

const PARTICLES: u32 = 600;
const MAX_RADIUS: f32 = 150.;
const DELTA_TIME: f32 = 0.1;
const REPELL_THRESHOLD: f32 = 20.;
const FRICTION: f32 = 0.1;
const TYPES: usize = 7;
const RADIUS: f32 = 3.;

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
enum Color {
    RED,
    BLUE,
    GREEN,
    WHITE,
    YELLOW,
    PURPLE,
    PINK,
}

impl Color {
    fn to_macroquad_color(&self) -> macroquad::prelude::Color {
        match self {
            Color::RED => RED,
            Color::BLUE => BLUE,
            Color::GREEN => GREEN,
            Color::WHITE => WHITE,
            Color::YELLOW => YELLOW,
            Color::PURPLE => PURPLE,
            Color::PINK => PINK,
        }
    }

    fn to_index(&self) -> usize {
        match self {
            Color::RED => 0,
            Color::BLUE => 1,
            Color::WHITE => 2,
            Color::YELLOW => 3,
            Color::GREEN => 4,
            Color::PURPLE => 5,
            Color::PINK => 6,
        }
    }
}

impl Distribution<Color> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Color {
        match rng.gen_range(0..7) {
            0 => Color::RED,
            1 => Color::BLUE,
            2 => Color::WHITE,
            3 => Color::YELLOW,
            4 => Color::PURPLE,
            5 => Color::PINK,
            _ => Color::GREEN,
        }
    }
}

#[derive(Clone)]
struct Particle {
    pos: Point,
    vel: Point,
    color: Color,
}

impl Particle {
    fn draw(&self) {
        draw_circle(
            self.pos.x,
            self.pos.y,
            RADIUS,
            self.color.to_macroquad_color(),
        )
    }

    fn update(&mut self, current_positions: &Vec<Particle>, rules: &Vec<Vec<f32>>) {
        let this_pos = self.pos;
        for Particle { pos, vel: _, color } in current_positions.iter() {
            let mut dx = pos.x - this_pos.x;
            dx = if dx.abs() < screen_width() - dx.abs() {
                dx
            } else {
                -1.0 * dx.signum() * (screen_width() - dx.abs())
            };
            let mut dy = pos.y - this_pos.y;
            dy = if dy.abs() < screen_height() - dy.abs() {
                dy
            } else {
                -1.0 * dy.signum() * (screen_height() - dy.abs())
            };
            let dist = (dx.powi(2) + dy.powi(2)).sqrt();
            if dist > -0.001 && dist < 0.001 {
                continue;
            }
            let force = force(
                dist,
                rules[self.color.to_index()][color.to_index()],
                Point { x: dx, y: dy },
            );
            self.vel = self.vel + force * DELTA_TIME;
        }
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

fn setup() -> (Vec<Particle>, Vec<Vec<f32>>) {
    let mut particles = Vec::new();
    for _ in 0..PARTICLES {
        particles.push(Particle {
            pos: Point {
                x: rand::gen_range(0., screen_width()),
                y: rand::gen_range(0., screen_height()),
            },
            vel: Point { x: 0., y: 0. },
            color: random(),
        })
    }
    println!(
        "Green is: {}, Red is: {}, Blue is: {},White is: {} ,Yellow is: {}, Purple is {}, Pink is {}",
        Color::GREEN.to_index(),
        Color::RED.to_index(),
        Color::BLUE.to_index(),
        Color::WHITE.to_index(),
        Color::YELLOW.to_index(),
        Color::PURPLE.to_index(),
        Color::PINK.to_index()
    );

    let mut rules = Vec::new();
    for c1 in 0..TYPES {
        let mut rule_row = Vec::new();
        for c2 in 0..TYPES {
            let r = random::<f32>() * 2. - 1.;
            rule_row.push(r);

            println!("{} is attracted to {} by {}", c1, c2, r);
        }
        rules.push(rule_row);
    }
    return (particles, rules);
}

#[macroquad::main("BasicShapes")]
async fn main() {
    let (mut particles, rules) = setup();
    loop {
        let current_particle_positions = particles.clone();
        clear_background(BLACK);
        // for particle in particles.iter_mut() {
        // particle.update(&current_particle_positions, &rules);
        // particle.draw();
        // }
        // draw_circle(10., 10., 5., WHITE);
        next_frame().await
    }
}
