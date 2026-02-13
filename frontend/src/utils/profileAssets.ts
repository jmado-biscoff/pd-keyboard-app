// Student profile images
import student_cat from "@/assets/profiles/student_profile/student_cat.jpg";
import student_dog from "@/assets/profiles/student_profile/student_dog.jpg";
import student_duck from "@/assets/profiles/student_profile/student_duck.jpg";
import student_frog from "@/assets/profiles/student_profile/student_frog.jpg";
import student_owl from "@/assets/profiles/student_profile/student_owl.jpg";
import student_rabbit from "@/assets/profiles/student_profile/student_rabbit.jpg";

// Teacher profile images
import teacher_chicken from "@/assets/profiles/teacher_profile/teacher_chicken.jpg";
import teacher_cow from "@/assets/profiles/teacher_profile/teacher_cow.jpg";
import teacher_crow from "@/assets/profiles/teacher_profile/teacher_crow.jpg";
import teacher_rhino from "@/assets/profiles/teacher_profile/teacher_rhino.jpg";
import teacher_sheep from "@/assets/profiles/teacher_profile/teacher_sheep.jpg";
import teacher_spider from "@/assets/profiles/teacher_profile/teacher_spider.jpg";

export const studentProfiles: { key: string; src: string }[] = [
  { key: "student_cat.jpg", src: student_cat },
  { key: "student_dog.jpg", src: student_dog },
  { key: "student_duck.jpg", src: student_duck },
  { key: "student_frog.jpg", src: student_frog },
  { key: "student_owl.jpg", src: student_owl },
  { key: "student_rabbit.jpg", src: student_rabbit },
];

export const teacherProfiles: { key: string; src: string }[] = [
  { key: "teacher_chicken.jpg", src: teacher_chicken },
  { key: "teacher_cow.jpg", src: teacher_cow },
  { key: "teacher_crow.jpg", src: teacher_crow },
  { key: "teacher_rhino.jpg", src: teacher_rhino },
  { key: "teacher_sheep.jpg", src: teacher_sheep },
  { key: "teacher_spider.jpg", src: teacher_spider },
];

/** Map from filename key to the Vite-resolved asset URL */
const profileMap: Record<string, string> = {};
for (const p of [...studentProfiles, ...teacherProfiles]) {
  profileMap[p.key] = p.src;
}

/** Resolve a stored filename (e.g. "student_frog.jpg") to its imported asset URL */
export function resolveProfileImage(key: string): string | undefined {
  return profileMap[key];
}

/** Get a random profile image filename for the given role */
export function getRandomProfile(role: "student" | "teacher"): string {
  const list = role === "student" ? studentProfiles : teacherProfiles;
  const idx = Math.floor(Math.random() * list.length);
  return list[idx].key;
}

/** Get the profile list for a given role */
export function getProfilesForRole(role: "student" | "teacher") {
  return role === "student" ? studentProfiles : teacherProfiles;
}
