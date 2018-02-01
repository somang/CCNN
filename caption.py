
class Caption:
  sequence = 0
  start = 0
  end = 0
  txt = ""

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
    return "{0}: {1}->{2}\n{3}".format(sequence, start, end, txt)


class CaptionTime:
  def __init__(self, hour, minute, second):
    self.hour = int(hour)
    self.minute = int(minute)
    if len(second.split(",")) == 2:
      self.second = second.split(",")[0]
      self.microsecond = second.split(",")[1]
    else:
      self.second = int(second)

  def __str__(self):
    return "{0}:{1}:{2},{3}".format(self.hour, self.minute, self.second, self.microsecond)