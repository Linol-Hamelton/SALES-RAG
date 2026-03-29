function pad(value) {
  return String(value).padStart(2, "0");
}

function formatWithTimezone(date) {
  const year = date.getFullYear();
  const month = pad(date.getMonth() + 1);
  const day = pad(date.getDate());
  const hours = pad(date.getHours());
  const minutes = pad(date.getMinutes());
  const seconds = pad(date.getSeconds());
  const offsetMinutes = -date.getTimezoneOffset();
  const sign = offsetMinutes >= 0 ? "+" : "-";
  const absoluteOffset = Math.abs(offsetMinutes);
  const offsetHours = pad(Math.floor(absoluteOffset / 60));
  const offsetMins = pad(absoluteOffset % 60);

  return `${year}-${month}-${day}T${hours}:${minutes}:${seconds}${sign}${offsetHours}:${offsetMins}`;
}

export default async function curdate() {
  const today = new Date();
  const firstDay = new Date(today.getFullYear() - 5, today.getMonth(), 1, 0, 0, 0, 0);
  const lastDay = new Date(today);

  return {
    firstDay: formatWithTimezone(firstDay),
    lastDay: formatWithTimezone(lastDay),
  };
}
