
class Caption:
  def __init__(self):
    self.sequence = 0
    self.start = 0
    self.end = 0
    self.txt = ""

  def setValues(self, seq, start_time, end_time, text):
    self.sequence = seq
    self.start = start_time
    self.end = end_time
    self.txt = text

  def setSeq(self, seq):
    self.sequence = seq
  
  def setTime(self, timeline):
    # parse time in format of 
    # 00:00:00,220 --> 00:00:04,270
    # into start and end time
    time_list = timeline.split("-->")
    st, et = time_list[0].strip(), time_list[1].strip()
    self.start = CaptionTime(st.split(":")[0], st.split(":")[1], st.split(":")[2])
    self.end = CaptionTime(et.split(":")[0], et.split(":")[1], et.split(":")[2])

  def setText(self, text):
    self.txt = text

  def __str__(self):
    return "{0}:\t{1}->{2}\n{3}".format(self.sequence, self.start, self.end, self.txt)

class CaptionTime:
  def __init__(self, hour, minute, second):
    self.hour = int(hour)
    self.minutes = int(minute)
    if len(second.split(",")) == 2:
      self.second = int(second.split(",")[0])
      self.milliseconds = int(second.split(",")[1])
    else:
      self.second = int(second)

  def to_ms(self):
    tmp_t = 0
    if self.milliseconds > 0:
      tmp_t += self.milliseconds
    tmp_t += (self.hour * 60 * 60 * 1000) + (self.minutes * 60 * 1000) + (self.second * 1000)
    return tmp_t

  def __str__(self):
    return "{0}:{1}:{2},{3}".format(self.hour, self.minutes, self.second, self.milliseconds)